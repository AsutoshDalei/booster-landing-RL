/**
 * rl_controller.js
 * Interface for ONNX Runtime to drive the rocket.
 */

const ort = window.ort;

export const RLController = {
    session: null,
    modelLoaded: false,
    isBusy: false,

    // Circuit Breaker
    errorCount: 0,
    MAX_ERRORS: 5,

    // Constants matching Python Training Environment
    NORM_X: 200.0,
    NORM_Y: 400.0,
    NORM_VX: 50.0,
    NORM_VY: 50.0,
    NORM_ANG_VEL: 10.0,

    // Action scaling
    ACTION_GIMBAL_SCALE: 0.6, // radians

    async init() {
        try {
            // Log the current env config to verify index.html settings
            console.log("RL Init - ORT Env:", JSON.stringify(ort.env));

            // Re-enforce strictly if not set
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = false;
            ort.env.wasm.proxy = false;

            // Path to the .onnx file
            const modelPath = './rl/falcon_ppo_best.onnx';

            // Configure session options with explicit threading constraints
            const sessionOptions = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                intraOpNumThreads: 1,
                interOpNumThreads: 1,
                executionMode: 'sequential' // disable parallel execution
            };

            this.session = await ort.InferenceSession.create(modelPath, sessionOptions);
            this.modelLoaded = true;
            console.log("RL Controller: ONNX Model Loaded Successfully");
        } catch (e) {
            console.error("RL Controller: Failed to load ONNX model", e);
            this.modelLoaded = false;
        }
    },

    /**
     * Get control actions from the RL model.
     * @param {Object} rocket - Rocket state
     * @param {Object} pad - Pad state
     * @returns {Object|null} - { throttle, engineGimbal, rcsLeft, rcsRight } or null if not ready
     */
    async getAction(rocket, pad) {
        // Early return if already busy or session not ready
        if (this.isBusy || !this.modelLoaded || !this.session) {
            return null;
        }

        // Set busy flag immediately to prevent concurrent calls
        this.isBusy = true;

        try {
            const normX = (rocket.x - pad.x) / this.NORM_X;
            const normY = (pad.y - rocket.y) / this.NORM_Y;
            const normVx = rocket.vx / this.NORM_VX;
            const normVy = rocket.vy / this.NORM_VY;

            // Wrap angle to [-pi, pi]
            let angle = rocket.angle;
            angle = (angle + Math.PI) % (2 * Math.PI) - Math.PI;
            const normAngle = angle / Math.PI; // Normalized -1 to 1

            const normAngVel = rocket.angularVelocity / this.NORM_ANG_VEL;

            // DEBUG: Check inputs
            if (Math.random() < 0.05) { // partial logging
                console.log(`RL Inputs -> Y:${normY.toFixed(2)} (${rocket.y.toFixed(0)}), VY:${normVy.toFixed(2)} (${rocket.vy.toFixed(0)}), Ang:${normAngle.toFixed(2)}`);
            }

            const inputTensor = new ort.Tensor(
                'float32',
                new Float32Array([normX, normY, normVx, normVy, normAngle, normAngVel]),
                [1, 6]
            );

            // Verify session is still valid before using
            if (!this.session || !this.session.inputNames || this.session.inputNames.length === 0) {
                throw new Error("Session is invalid or not properly initialized");
            }

            // 2. Run Inference
            const inputName = this.session.inputNames[0];
            const result = await this.session.run({ [inputName]: inputTensor });

            // Output name? Usually output.1 or similar.
            const outputName = this.session.outputNames[0];
            const outputData = result[outputName].data; // Float32Array

            // 3. Decode Action (Match Python step)
            // The model outputs the MEAN of the Normal distribution (unbounded)
            // During training: action = tanh(sample(Normal(mean, std)))
            // During inference: we use tanh(mean) to get deterministic action in [-1, 1]
            // NOTE: If the model was exported with tanh included, outputs are already in [-1, 1]
            // We detect this by checking if values are outside [-1.1, 1.1] range
            
            // Output 0: Throttle Action
            let rawThrottle = outputData[0];
            let actionThrottle;
            // Check if this is a raw mean (outside [-1.1, 1.1]) or already tanh'd
            const isRawMean = Math.abs(rawThrottle) > 1.1;
            if (isRawMean) {
                // Old model: Raw mean - apply tanh
                // This should not happen with the new model that includes tanh
                actionThrottle = Math.tanh(rawThrottle);
                if (Math.random() < 0.1) {
                    console.warn("RL: Using old model format (raw mean detected). Consider re-exporting with tanh.");
                }
            } else {
                // New model: Already in [-1, 1] range - use directly (model was exported with tanh)
                actionThrottle = rawThrottle;
            }
            // Transform from [-1, 1] to [0, 1] for throttle
            // Python: throttle = (action + 1) / 2
            const throttle = (actionThrottle + 1.0) / 2.0;

            // Output 1: Gimbal Action
            let rawGimbal = outputData[1];
            let actionGimbal;
            if (Math.abs(rawGimbal) > 1.1) {
                actionGimbal = Math.tanh(rawGimbal);
            } else {
                actionGimbal = rawGimbal;
            }
            // Python: gimbal = action * 0.6
            const gimbal = actionGimbal * this.ACTION_GIMBAL_SCALE;

            // Output 2: RCS Action
            let rawRcs = outputData[2];
            let actionRcs;
            if (Math.abs(rawRcs) > 1.1) {
                actionRcs = Math.tanh(rawRcs);
            } else {
                actionRcs = rawRcs;
            }

            // Convert RCS continuous to discrete boolean for JS State
            // Deadband 0.3 matches Python
            let rcsLeft = false;
            let rcsRight = false;
            if (actionRcs < -0.3) {
                // Push Right (Positive X force) -> Thruster on Left fires? 
                rcsLeft = true;
            } else if (actionRcs > 0.3) {
                // Push Left (Negative X force) -> Thruster on Right fires?
                rcsRight = true;
            }

            // Debug logging
            if (Math.random() < 0.05) {
                console.log(`RL Output -> Action:${actionThrottle.toFixed(2)}, Throttle:${throttle.toFixed(3)}, Gimbal:${gimbal.toFixed(3)}, RCS:${actionRcs.toFixed(2)}`);
            }

            // Clamp throttle to [0, 1] range
            let finalThrottle = Math.max(0, Math.min(1, throttle));
            
            // IMPORTANT: Apply deadband to prevent tiny throttle values from triggering MIN_THROTTLE in physics
            // MIN_THROTTLE (0.4) means even throttle=0.01 becomes power=0.406, causing unwanted ascent
            // Only values above 0.05 will actually activate the engine (matches physics.js THROTTLE_DEADBAND)
            if (finalThrottle < 0.05) {
                finalThrottle = 0.0;
            }

            return {
                throttle: finalThrottle,
                engineGimbal: gimbal,
                rcsLeft: rcsLeft,
                rcsRight: rcsRight
            };

        } catch (e) {
            console.error("RL Inference Error:", e);

            // Circuit Breaker Logic
            this.errorCount++;

            // If it's a fatal session error, kill it immediately to stop loop
            const isFatal = e.message.includes("Session") || e.message.includes("mismatch");

            if (this.errorCount > this.MAX_ERRORS || isFatal) {
                console.warn("RL Controller: Fatal/Too many errors. Disabling RL Model.");
                this.modelLoaded = false;
                this.session = null;
            }

            return 'RL_ERROR'; // Explicit error signal
        } finally {
            // Always reset busy flag, even on error
            this.isBusy = false;
        }
    }
};
