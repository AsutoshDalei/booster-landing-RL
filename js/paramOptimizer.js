/**
 * paramOptimizer.js
 * Evolutionary algorithm for optimizing landing parameters
 */

// Canvas dimensions for headless simulations
let CANVAS_WIDTH = 800;
let CANVAS_HEIGHT = 600;

export function setCanvasDimensions(width, height) {
    CANVAS_WIDTH = width;
    CANVAS_HEIGHT = height;
}

// Parameter bounds for optimization
const PARAM_BOUNDS = {
    ignitionAltitude: { min: 100, max: 600 },
    gimbalKp: { min: 0.5, max: 5.0 },
    gimbalKd: { min: 0.1, max: 3.0 },
    throttleKp: { min: -0.2, max: -0.01 },
    throttleKi: { min: -0.05, max: 0.0 },
    throttleKd: { min: -0.1, max: -0.001 }
};

// Algorithm parameters
const POPULATION_SIZE = 30;
const GENERATIONS = 15;
const MUTATION_RATE = 0.15;
const CROSSOVER_RATE = 0.7;
const ELITE_COUNT = 6;
const MAX_SIM_TIME = 30; // seconds

/**
 * Individual represents a set of parameters
 */
class Individual {
    constructor() {
        this.params = {
            ignitionAltitude: 0,
            gimbalKp: 0,
            gimbalKd: 0,
            throttleKp: 0,
            throttleKi: 0,
            throttleKd: 0
        };
        this.fitness = -Infinity;
    }

    static random() {
        const ind = new Individual();
        for (const key in PARAM_BOUNDS) {
            const bounds = PARAM_BOUNDS[key];
            ind.params[key] = bounds.min + Math.random() * (bounds.max - bounds.min);
        }
        return ind;
    }

    mutate() {
        for (const key in this.params) {
            if (Math.random() < MUTATION_RATE) {
                const bounds = PARAM_BOUNDS[key];
                const range = bounds.max - bounds.min;
                const mutation = (Math.random() - 0.5) * range * 0.2; // 20% of range
                this.params[key] = Math.max(bounds.min, Math.min(bounds.max, this.params[key] + mutation));
            }
        }
    }

    static crossover(parent1, parent2) {
        const child = new Individual();
        for (const key in parent1.params) {
            // Uniform crossover
            child.params[key] = Math.random() < 0.5 ? parent1.params[key] : parent2.params[key];
        }
        return child;
    }
}

/**
 * Run a headless simulation with given parameters
 */
async function runSimulation(params, simState, stepPhysics, Guidance, Autopilot, resetRocket) {
    // Reset state
    resetRocket(simState, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Apply parameters
    simState.tuning.ignitionAltitude = params.ignitionAltitude;
    simState.tuning.gimbalKp = params.gimbalKp;
    simState.tuning.gimbalKd = params.gimbalKd;
    simState.tuning.throttleKp = params.throttleKp;
    simState.tuning.throttleKi = params.throttleKi;
    simState.tuning.throttleKd = params.throttleKd;
    
    // Reset autopilot
    Autopilot.reset();
    
    const FIXED_DT = 1.0 / 60.0;
    let time = 0;
    const maxSteps = Math.floor(MAX_SIM_TIME / FIXED_DT);
    
    for (let step = 0; step < maxSteps; step++) {
        Guidance.update(simState, FIXED_DT);
        Autopilot.update(simState, FIXED_DT);
        stepPhysics(simState, FIXED_DT);
        
        time += FIXED_DT;
        
        // Check for landing result
        if (simState.guidance.landingResult !== null) {
            return {
                success: simState.guidance.landingResult === 'SUCCESS',
                time: time,
                timeout: false,
                state: JSON.parse(JSON.stringify(simState))
            };
        }
        
        // Yield occasionally to prevent blocking
        if (step % 60 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    return { success: false, time: MAX_SIM_TIME, timeout: true, state: JSON.parse(JSON.stringify(simState)) };
}

/**
 * Evaluate fitness of a parameter set
 */
async function evaluateParameters(params, numTests, simStateFactory, stepPhysics, Guidance, Autopilot, resetRocket) {
    let totalFitness = 0;
    let successes = 0;
    
    for (let test = 0; test < numTests; test++) {
        // Create a fresh state copy for this test
        const testState = simStateFactory();
        
        const result = await runSimulation(params, testState, stepPhysics, Guidance, Autopilot, resetRocket);
        
        const r = result.state.rocket;
        const pad = result.state.pad;
        let fitness = 0;
        
        if (result.success) {
            successes++;
            // Success - high fitness based on quality
            const padTop = pad.y - pad.height / 2;
            const rocketBottom = r.y + r.height / 2;
            const altitude = Math.max(0, padTop - rocketBottom);
            const xError = Math.abs(r.x - pad.x);
            const angleError = Math.abs(r.angle);
            const speedError = Math.abs(r.vy);
            
            // Reward: low errors, fast landing
            fitness = 2000 - (xError * 20) - (angleError * 200) - (speedError * 50) - (result.time * 2);
        } else {
            // Failure - negative fitness based on how close we got
            const padTop = pad.y - pad.height / 2;
            const rocketBottom = r.y + r.height / 2;
            const altitude = Math.max(0, padTop - rocketBottom);
            const xError = Math.abs(r.x - pad.x);
            const angleError = Math.abs(r.angle);
            const speedError = Math.abs(r.vy);
            
            // Penalize: distance from success, but reward getting close
            let partialCredit = 0;
            if (altitude < 10 && xError < 50 && angleError < 0.2) {
                partialCredit = 100; // Partial success bonus
            }
            
            fitness = partialCredit - altitude - (xError * 3) - (angleError * 100) - (speedError * 20) - 1000;
        }
        
        totalFitness += fitness;
    }
    
    return {
        fitness: totalFitness / numTests,
        successRate: successes / numTests
    };
}

/**
 * Parameter Optimizer using Evolutionary Algorithm
 */
export class ParameterOptimizer {
    constructor(simStateFactory, stepPhysics, Guidance, Autopilot, resetRocket) {
        this.simStateFactory = simStateFactory;
        this.stepPhysics = stepPhysics;
        this.Guidance = Guidance;
        this.Autopilot = Autopilot;
        this.resetRocket = resetRocket;
        
        this.population = [];
        this.generation = 0;
        this.bestFitness = -Infinity;
        this.bestIndividual = null;
        this.isRunning = false;
        this.onProgress = null;
    }
    
    stop() {
        this.isRunning = false;
    }
    
    async initialize() {
        this.population = [];
        for (let i = 0; i < POPULATION_SIZE; i++) {
            this.population.push(Individual.random());
        }
        this.generation = 0;
        this.bestFitness = -Infinity;
        this.bestIndividual = null;
    }
    
    async evolve(numTestsPerIndividual = 3) {
        this.isRunning = true;
        await this.initialize();
        
        for (let gen = 0; gen < GENERATIONS && this.isRunning; gen++) {
            this.generation = gen;
            
            // Evaluate all individuals
            for (let i = 0; i < this.population.length && this.isRunning; i++) {
                const individual = this.population[i];
                
                const result = await evaluateParameters(
                    individual.params,
                    numTestsPerIndividual,
                    this.simStateFactory,
                    this.stepPhysics,
                    this.Guidance,
                    this.Autopilot,
                    this.resetRocket
                );
                
                individual.fitness = result.fitness;
                
                // Track best
                if (individual.fitness > this.bestFitness) {
                    this.bestFitness = individual.fitness;
                    this.bestIndividual = {
                        params: { ...individual.params },
                        fitness: individual.fitness
                    };
                }
                
                // Progress callback
                if (this.onProgress) {
                    const progress = ((gen * POPULATION_SIZE + i + 1) / (GENERATIONS * POPULATION_SIZE)) * 100;
                    this.onProgress({
                        generation: gen + 1,
                        individual: i + 1,
                        progress: progress,
                        bestParams: this.bestIndividual ? this.bestIndividual.params : null,
                        bestFitness: this.bestFitness
                    });
                }
            }
            
            if (!this.isRunning) break;
            
            // Sort by fitness
            this.population.sort((a, b) => b.fitness - a.fitness);
            
            // Create next generation
            const nextGen = [];
            
            // Elitism: keep best individuals
            for (let i = 0; i < ELITE_COUNT; i++) {
                const elite = new Individual();
                elite.params = { ...this.population[i].params };
                elite.fitness = this.population[i].fitness;
                nextGen.push(elite);
            }
            
            // Generate rest through crossover and mutation
            while (nextGen.length < POPULATION_SIZE && this.isRunning) {
                // Tournament selection
                const parent1 = this.tournamentSelect();
                const parent2 = this.tournamentSelect();
                
                let child;
                if (Math.random() < CROSSOVER_RATE) {
                    child = Individual.crossover(parent1, parent2);
                } else {
                    // Clone parent1 as a new Individual
                    child = new Individual();
                    child.params = { ...parent1.params };
                }
                
                child.mutate();
                nextGen.push(child);
            }
            
            this.population = nextGen;
        }
        
        this.isRunning = false;
        return this.bestIndividual ? this.bestIndividual.params : null;
    }
    
    tournamentSelect(tournamentSize = 3) {
        let best = null;
        for (let i = 0; i < tournamentSize; i++) {
            const candidate = this.population[Math.floor(Math.random() * this.population.length)];
            if (!best || candidate.fitness > best.fitness) {
                best = candidate;
            }
        }
        return best;
    }
}
