# Sensor Fusion (part 1 of ?)

## Intro

This blog post is about sensor fusion. You might think you don't know what that means, but don't worry, you do. You just don't know you do. It's something you do all the time, as part of your daily life. In fact, a lot of it is done by your nervous system autonomously, so you might not even notice that it's there unless you look for it. Take, for example, the sense of sight and the sense of smell. It is well known that the active aromatic compound in both parmesan cheese and vomit is the same. So, if we were to rely on smell alone, we would not be able to distinguish them. However, with adding the sense of sight to the equation, our brain resolves the dilemma instantly and decisively.

Ok, maybe that's too much detail, or maybe one too many vomit references. Let me try and make more of an outline and less of a run-off sentence.

## Sensing Number of People in a Room

We'll use the example of trying to figure out, using simple sensors, how many people are in a room at a given time. We'll start slow while introducing important concepts, and make our way to a practical application.

In our example, we have a conference room that seats somewhere between zero and 20 people. We want to install sensors that help us know how many people there are in the room at a given time. This might be because we want to optimize the use of the room, or to figure out when the room is least busy, and schedule cleaning during those time.

### CO2 Sensors

A simple way to get a (very rough) headcount in a room is to measure the level of carbon dioxide in a room. Since people breathe out more CO2 than they breathe in, a higher concentration of CO2 in a room usually correlates with more people. How much, however, depends a lot on the room, and maybe on other things, like light or time of day, especially if there are living plants in the room. For this tutorial though, let's simply assume the only CO2 sources in the room are people.

Under this assumption we go and install a CO2 sensor in the middle of the room. We also design and run a simple experiment, where we take readings of both the CO2 level and the number of people present at that time in the room. Let's say we get a figure like this ![CO2 sensor curve](co2curve.jpg)

### Sensor Models

The above figure helps up build a _sensor model_ for our particular sensor/room. A simple model would be to fit a line to the points in the above figure. That tells us, for every given reading, how many people we expect to find in the room. We could stop here, report that number of people, and call it done. But we're no fools, we see there's a LOT of noise in the reading. So let's also calculate the standard error for our fit, to get a rough estimate of how good our model is.

All right, so now we have a model that says "when the CO2 sensor tells us it's reading X ppm of CO2, we can expect there to be Y±e". In fact, according to this model, each sensor reading can be interpreted as a _probability distribution_, like in the following figure ![First Reading](reading1.jpg). In the figure we show a reading of 0.005 on the sensor. According to our earlier experiments, this means that the most likely number of occupants in the room is 3\. However, there's a non-zero chance that there are zero occupants, or 20\. The curve shows this. In fact, this single measurement is completely useless if we care to know if the room is empty or not, since the likelihood there are 3 people in the room is only slightly higher than the likelihood it is empty.

ASIDE: If you think that this is a simplistic model, that's because it is. This is more or less the smallest building block of sensor fusion theory, and like any "smallest block" it has its problems. For example, you might have noticed that there's a non-zero probability there are negative occupants in the room. That's not a real issue though, as we can think of that as "the reading was so low, it was even _more_ unlikely there was anyone there"

### Multiple readings

Now that we have a sensor model, let's actually get some data! For this part of the tutorial, let's assume we know the room is used for day-long meetings, so that the number of people in the room is constant throughout the day. Again, don't worry if this seems simplistic: we have to start somewhere. Now, we already have a sensor reading (the one from fig. 2), so let's grab one more ![Second Reading](reading2.jpg). This one is centered at 5 people and has, unsurprisingly, the same standard deviation as the first reading---this is a side effect of our model. Now comes the fun part: we merge the two readings into one probability curve ![Added readings](fusion1.jpg). Mathematically, we averaged the two curves by adding them and dividing the result by two. Intuitively, we moved our "best guess" to 4 people, but also increased our confidence in the the result. Since a CO2 sensor can give a measurement several times every second, we could repeat this thousands of times throughout the day to generate our curve from all those measurements. However, if our assumption is correct and the number of people in the room stays constant, we'll soon find that adding new measurements won't change the estimate significantly.

### Multiple sensors

We can apply the process in the previous section to multiple sensors that measure the same thing. For example, let's say we want to install a temperature sensor, whose measurements also (loosely) correlate with the number of people in the room. After doing the required experiments, the new sensor curve looks like this (shown next to the original sensor's curve, for comparison) ![Two sensor curve](fusion2.jpg).

Now, each measurement throughout the day could come from one sensor, or the other. Since we know how much to trust each sensor, we use the same method as before to fuse them together. The math stays relatively simple, since we're still averaging Gaussians: we add up all the curves and normalize the result.

### Moving targets

Until now we assumed the number of people in the room stays constant. That's clearly a bad assumption, because people like to move around from time to time. Fortunately, the previous approach can be easily adapted to moving targets. In order to do this, we need to introduce the concept of time between measurements, and also build a model of human behavior between measurements. In fact, we already _have_ a model for the human behavior between measurements: we model the number of people as non-changing. Let's instead assume that people entering and leaving the room is a Poisson process, with one person entering and one person leaving the room every 10 minutes. So, for example, say at 9am we know for a fact there were 10 people in the room---our probability curve is a _very_ sharp Gaussian centered at 6\. Every second without looking in the room our confidence declines. So, at 9:10 we're not really sure if there are 5, or 6 or 7 people in the room. In fact, we can describe this mathematically:

**EQUATION HERE**

If we let enough time pass (about an hour) we won't be able to say pretty much anything about the state of the room. However, let's say we take a measurement from our first CO2 sensor at 9:30am. At this point in time we have a _prior_ estimate, as well as the measurement estimate. They look like this ![Prior and update](priorandupdate.jpg). We update our prior with the new data, which instantly narrows our Gaussian. While we wait for the next measurement our Gaussian spreads out again, according to our human model. In the end we get a filtered version of the signal, translated from CO2 levels to number of people in the room. We also get a confidence interval at every point in time.

You can probably already see how this would work in the case of both the CO2 and the temperature sensor: simply update the prior (the current guess) with the new mean and standard deviation from the sensor model. By transforming everything into the variable we want to measure, we've transformed the problem of sensor data fusion into a problem of sensor modeling.

### Better Models

What we did in the previous section is a well know, and quite brilliant method, called _Kalman filtering_. It's one of the technological advances that helped Apollo 11 get to the moon, and variants of it are used nowadays on pretty much any non-trivial control system. An old joke said that every time late Rudolf Kálmán got on a plane he would state his name and ask to be show "his filter".

Improvements on this approach are numerous, and I'm sure some of you are already thinking about what they might be. I'll list only a few:

- non-linear sensor models: we fit a polynomial or exponential curve to the experimental data
- non-constant sensor noise model: the standard deviation is different at different sensor readings, not constant across the range of readings
- time-varying sensor model: the way the sensors behave might change with time of day, or other factors (e.g. air conditioning)
- more complex human dynamics: the way people enter and leave a room is a lot more complicated than a Poisson process; we could, for example, take into account that people are more likely to enter and leave at full and half hours, when meetings usually start and end; also, depending on company culture and the function of the room, we might know that people are more likely to enter than leave in the morning and vice-versa in the afternoon.
- process noise: we have only assumed that the sensor is noisy, but that the underlying process is smooth; usually they are _both_ noisy, with different noise profiles (at the very least different standard deviations) and taking that into account can improve the filter.
- some other stuff I haven't thought of, but you probably have: add it to the comments!

### See also

I purposefully kept this post light, especially in terms of math. Just a quick look at the wikipedia page for Kalman filtering is enough to turn most people off the whole subject. I wanted to make sure that you first and foremost follow the concepts presented here intuitively, rather than mathematically.

However, I bet there are some of you who want more "meat" than what I've given here. I hear you. Consider this an appetizer. I'll let you do your own cooking for the main meal, but I'll give a list of potential ingredients:

- [A succint technical version of this presentation, for those familiar with complex statistics notation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&ved=0ahUKEwjih5yV3_3OAhUh7YMKHagpB3sQFgg1MAM&url=http%3A%2F%2Fwww.stats.ox.ac.uk%2F~steffen%2Fteaching%2Fbs2HT9%2Fkalman.pdf&usg=AFQjCNH2QTxgfPjpvlUa4zLtiICIS_JzZQ&sig2=Zv2TImCe80IoavJx8_jI6A)

**list of links goes here**
