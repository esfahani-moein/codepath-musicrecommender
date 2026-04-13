# Reflection: Profile Comparison & Evaluation Notes

## High-Energy Pop vs. Chill Lofi

The High-Energy Pop profile surfaces "Sunrise City" and "Gym Hero" — both pop genre, high energy. The Chill Lofi profile surfaces "Library Rain" and "Midnight Coding" — both lofi, chill, low energy, with acoustic bonus. The key difference is energy: the pop profile targets 0.85 while lofi targets 0.35, and the energy similarity score (worth up to 1.5 points) drives this split. This makes sense because energy is the most heavily weighted numerical feature — it effectively separates "pump-up" tracks from "wind-down" tracks.

## Deep Intense Rock vs. Acoustic Jazz Lover

The Rock profile gets "Storm Runner" (#1) — rock/intense/high-energy, a perfect match. The Jazz profile gets "Coffee Shop Stories" (#1) — jazz/relaxed/low-energy/acoustic, also perfect. What's interesting is that both profiles share a similar energy gap from the center, but in opposite directions. The jazz profile benefits from the acoustic bonus (+0.5) which the rock profile never triggers because rock listeners don't flag `likes_acoustic=True`. This shows how a boolean flag can create an asymmetric advantage — acoustic-leaning users get extra points that non-acoustic users can never earn, even from songs with moderate acousticness.

## Conflicting Profile vs. All Others

The Conflicting Profile (lofi genre + intense mood + high energy) is the most revealing. It asks for something that barely exists in the catalog: a high-energy lofi song with an intense mood. The system resolves this by falling back on the genre match (+2.0), pushing low-energy lofi songs to the top even though they contradict the energy and mood targets. "Storm Runner" (rock/intense) appears at #4 only because of the mood match (+1.0) and energy proximity. Compared to every other profile where the top result feels intuitive, the conflicting profile produces results that feel wrong — and that's the point. It exposes that the system doesn't understand the *relationship* between features, only their individual scores.

## Weight Experiment Impact

When we doubled the energy weight and halved the genre weight, the most notable change was "Rooftop Lights" (indie pop, happy, energy=0.76) jumping from #3 to #2 for the pop/happy profile, overtaking "Gym Hero" (pop, intense, energy=0.93). This makes intuitive sense: "Rooftop Lights" matches the user's mood (happy) and has closer energy to the target (0.80 vs 0.93), while "Gym Hero" only matches genre. The original weights masked this because the genre bonus was large enough to overcome the mood mismatch. In a real system, this kind of weight tuning would determine whether users discover mood-appropriate songs outside their usual genre or stay trapped in a genre bubble.
