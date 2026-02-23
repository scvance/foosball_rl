"""
demo_shootout.py

Watch the ShootoutSimCore run a few episodes with random paddle actions.
Run from the project root:

    python demo_shootout.py

Controls (PyBullet GUI):
    Mouse-drag     — orbit camera
    Scroll wheel   — zoom
    Ctrl+drag      — pan
"""

import random
import sys
import time

sys.path.insert(0, "foosball_envs")

from ShootoutSimCore import _ShootoutSimCore

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_EPISODES   = 20
MAX_STEPS      = 600         # steps per episode before forced reset
SIM_HZ         = 200         # physics steps per second
CTRL_HZ        = 30          # control decisions per second
SIM_PER_CTRL   = SIM_HZ // CTRL_HZ   # = 8 sub-steps per control tick
REAL_TIME      = True        # sleep to approximate real-time playback
SEED           = 42

# Alternate spawn modes each episode
SPAWN_MODES = ["fire_home", "fire_away", "corner"]

# ── Random action helpers ──────────────────────────────────────────────────────

def random_handle(limits, rng):
    """Random position target within joint limits."""
    return rng.uniform(limits.lower, limits.upper)

def random_paddle_vel(max_vel, rng):
    """Random angular velocity: full spin or hold."""
    roll = rng.random()
    if roll < 0.15:          # hold
        return 0.0
    elif roll < 0.55:        # spin forward
        return rng.uniform(5.0, max_vel)
    else:                    # spin backward
        return rng.uniform(-max_vel, -5.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    sim = _ShootoutSimCore(
        use_gui=True,
        time_step=1.0 / SIM_HZ,
        seed=SEED,
        num_substeps=1,   # one p.stepSimulation() per SIM_HZ tick
        ball_restitution=0.30,
        wall_restitution=0.85,
        paddle_restitution=0.85,
    )

    goals_home   = 0
    goals_away   = 0
    blocks_home  = 0
    blocks_away  = 0
    oob_count    = 0

    print("─" * 60)
    print("Shootout demo  |  close the PyBullet window to quit")
    print("─" * 60)

    try:
        for ep in range(NUM_EPISODES):
            sim.reset_robot()
            sim.remove_ball()

            # Pick a spawn mode
            mode = SPAWN_MODES[ep % len(SPAWN_MODES)]
            if mode == "fire_home":
                info = sim.spawn_ball_random_fire(target="home")
                label = "fire → home goal"
            elif mode == "fire_away":
                info = sim.spawn_ball_random_fire(target="away")
                label = "fire → away goal"
            else:
                info = sim.spawn_ball_corner_serve(side="random")
                label = f"corner serve ({info['side']})"

            print(f"\nEpisode {ep + 1:2d}/{NUM_EPISODES}  [{label}]")
            print(f"  spawn x={info['x_spawn']:.3f}  y={info['y_spawn']:.3f}"
                  f"  z={info['z_spawn']:.3f}")

            # Slowly-changing action targets (re-sampled every N ticks)
            action_interval = CTRL_HZ // 3   # change every ~330 ms
            home_handle_tgt  = random_handle(sim.handle_limits, rng)
            home_paddle_vel  = random_paddle_vel(sim.paddle_vel_cap, rng)
            away_handle_tgt  = random_handle(sim.opponent_handle_limits, rng)
            away_paddle_vel  = random_paddle_vel(sim.paddle_vel_cap, rng)

            terminated = False
            for tick in range(MAX_STEPS):
                # Resample actions periodically
                if tick % action_interval == 0:
                    home_handle_tgt = random_handle(sim.handle_limits, rng)
                    home_paddle_vel = random_paddle_vel(sim.paddle_vel_cap, rng)
                    away_handle_tgt = random_handle(sim.opponent_handle_limits, rng)
                    away_paddle_vel = random_paddle_vel(sim.paddle_vel_cap, rng)

                sim.apply_action_targets_dual(
                    home_handle_tgt, home_paddle_vel,
                    away_handle_tgt, away_paddle_vel,
                )
                sim.step_sim(SIM_PER_CTRL)

                # ── Event checks ─────────────────────────────────────────────
                goals = sim.check_goal_crossings_dual()
                if goals["home"]:
                    goals_home += 1
                    sim.update_scoreboard_text(goals_home, blocks_home, oob_count, highlight="goal")
                    print(f"  GOAL → home!   (step {tick})")
                    terminated = True
                elif goals["away"]:
                    goals_away += 1
                    sim.update_scoreboard_text(goals_away, blocks_away, oob_count, highlight="goal")
                    print(f"  GOAL → away!   (step {tick})")
                    terminated = True

                if not terminated:
                    blocks = sim.check_block_events_dual()
                    if blocks["home"]:
                        blocks_home += 1
                        sim.update_scoreboard_text(goals_home, blocks_home, oob_count, highlight="block")
                        print(f"  BLOCK (home)   (step {tick})")
                        terminated = False
                    elif blocks["away"]:
                        blocks_away += 1
                        sim.update_scoreboard_text(goals_away, blocks_away, oob_count, highlight="block")
                        print(f"  BLOCK (away)   (step {tick})")
                        terminated = False

                if not terminated:
                    oob, reason = sim.check_ball_out_of_bounds()
                    if oob:
                        oob_count += 1
                        sim.update_scoreboard_text(
                            goals_home + goals_away, blocks_home + blocks_away,
                            oob_count, highlight="out",
                        )
                        print(f"  OUT OF BOUNDS  ({reason}, step {tick})")
                        terminated = True

                if terminated:
                    time.sleep(0.4)   # brief pause so you can see the outcome
                    break

                if REAL_TIME:
                    time.sleep(SIM_PER_CTRL / SIM_HZ)

            if not terminated:
                print(f"  timeout after {MAX_STEPS} steps")

    except Exception as exc:
        print(f"\nDemo interrupted: {exc}")
    finally:
        sim.close()

    print("\n─" * 60)
    print(f"Final tally — goals home: {goals_home}  away: {goals_away}"
          f"  blocks home: {blocks_home}  away: {blocks_away}"
          f"  oob: {oob_count}")


if __name__ == "__main__":
    main()
