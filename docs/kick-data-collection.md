# Kick trajectory data collection

The default strategy can record every real `KickIntent` it produces, without
changing any movement or kick command. It uses only the public team-view ball
position and the Agent's own command, so it is suitable for development
simulation and does not rely on simulator-internal controls.

## Output

Structured logging is enabled by default. On activation, the Agent shell log
prints the ordinary structured-log path and a second line beginning with
`Kick calibration data path:`. The second path is normally a sibling file named
`kick_samples.team<N>.jsonl` under:

```text
/tmp/booster_agent/soccer_logs/<run-id>/
```

Each line is one completed kick sample. It includes the team and player IDs,
the requested kick direction/power, the kicker pose, referee context, and a
deduplicated sequence of public ball positions after the kick.

The sample ends when the ball stays within 2.5 cm over a 0.35 second window,
the ball data becomes stale, another own kick starts, eight seconds pass, or
the Agent closes. These are collection labels, not a physics claim: later
fitting should reject trajectories affected by robots, posts, boundaries, or
referee resets.

## Recommended collection run

1. Run the unmodified default strategy in Booster Studio.
2. Let it play normally; do not use non-public reset, movement, or ball-control
   interfaces to manufacture samples.
3. Copy the JSONL file after the run.
4. Keep only clear, uninterrupted trajectories for the first friction-model
   fit. Record rejected samples too; their end reason is useful context.

The first model should be evaluated offline against held-out trajectories before
it influences Chaser selection or movement targets.
