# Booster Landing Simulator

A physics-based 2D simulation of a rocket booster performing a propulsive landing, inspired by Falcon 9.

## Concept

The goal of this project is to simulate the complex control systems required to vertically land an orbital class rocket booster.

**Autopilot**: The simulation features a fully autonomous autopilot mode. It uses dual-loop PID controllers to maintain stability (gimbal and RCS) and a "suicide burn" algorithm to execute a precise, fuel-efficient landing burn at the last possible second.

## Motivation

I built this project primarily to experiment with **vibe coding** and to dive deeper into **Reinforcement Learning (RL)**.
*   **Vibe Coding**: Testing how AI tools can assist in rapid prototyping and development.
*   **Reinforcement Learning**: I am currently working on implementing an RL agent to teach the rocket to land itself from scratch, replacing the hardcoded PID logic. (Work in Progress)

## Web Deployment

You can try the simulation directly here:
**[Launch Simulator](https://asutoshdalei.github.io/booster-landing-RL/)**

## Running Locally (Docker)

You can run the project locally using Docker to ensure a consistent environment.

```bash
docker build -t rocket-sim .
docker run -p 8080:8080 -v $(pwd):/app rocket-sim
```
Then open `http://localhost:8080` in your browser.

## Controls

*   **P**: Toggle Autopilot (Watch it land itself!)
*   **R**: Reset Simulation
*   **W / S**: Throttle Up / Down
*   **A / D**: Gimbal Left / Right
*   **Q / E**: RCS Thrusters

Feel free to explore the code or tweak the parameters in the UI to see how the rocket behaves!

## Credits

*   Inspired by SpaceX validation footage and the work of [EmbersArc/gym-rocketlander](https://github.com/EmbersArc/gym-rocketlander).
*   Built largely through **AI collaboration** (Antigravity, ChatGPT) as a study in prompting and iterative design.
