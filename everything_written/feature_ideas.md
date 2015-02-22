# please insert good ideas for features

## first order features
x,y - thanks to rotation and trimming uninformative

## second order features
*between two points in time*
1. acceleration
2. speed
3. angle (- this is randomized for anonymity of the driver)
4. revisit - similar x,y visited again? (- first/ last few seconds are removed for anonymity)
5. deceleration/breaking (should be treated differently than acceleration)


## third order features

*calculated from second order features*

1. mean speed, accelleration, angle(?)

2. curving - acceleration given angle, speed given angle, maybe classify angle in light, medium and sharp curves depending on duration, angle itself, ...?

3. takes breaks every XXX minutes? Only on long trips? (check if long periods of stasis lead to the rest of the journey being counted as another trip)

4. length of epochs of (relatively) constant speed (i.e. drives on highway, stays there for 50 kilometers then leaves vs stop and go)

5. Try PCA over all ways to see if a "typical trip" can be inferred

6. ratio of long trips/short trips

7. behaviour on 90° curves // how many "long" 90° curves vs "short" 90°curves (classify as lives in city/ in rural area)

8. distance from origin during each timestep as measure of way finding (relatively straigt to target location: wants to get there. Staying within a certain circle around the origin: Probably delivery guy)

9. match trips to each other. If a similar trip comes up more than XX(e.g.10) times, we can set the probability of it belonging to the same driver to 1, since only "few" ways are from other random guys.

10. reverse start and endpoint of each trip and match to every other trip to see if driver took same way back (e.g. going to work and going back on the same way). Sliding window necessary because of moronic randomisation.

11. utilizing 10: ratio of "often used ways"/"rarely used ways", i.e. does the driver mainly use his car to go to work or does he make many unique trips

12. utilizing 10: mean speed on way "to work" vs "from work" (might indicate that e.g. on the way from work to home there is stop and go, while not on the reversed trip, which would be a very distinct pattern)

13. Maybe normalize all ways to "distance from origin = 1" or something. Not sure what can be accomplished by that.

14. "AUC", not really, but area in the "slope" that connects start and endpoints

15. correlation between angles can help to find same trips

16. abs(angle) correltaion to find return trips?

## thing that would be nice/ fourth order features

1. being able to infer if trip was at day or at night (no idea how to reliably do that)

2. does the driver try to conserve fuel

3. dies the driver prefer "fast" over "effective"(fuel consumption?)

4. does the driver use a gps (does he follow some kind of optimal trajectory? This might only be inferrable for trips that
occur often and are similar enough, also turning around a lot might be a valid feature-feature)

5.
## relatively hard classifiers

*features that will be relatively rare, but when they can be found they are reliable.*

**if they occur often enough, we can turn the unsupervised problem into a semi-supervised problem**

1. a way that comes up often should be of the original driver (e.g. going to work and home)

2. assuming that driver has shitty car: All trips that involve a certain level of acceleration must be from another car

3. assuming driver has heavy car: some trajectories not possible that might be possible with a smart(maybe this one is covered in too much noise)

4. upper speed limit? USA 80 mph max, so maybe not useful

