#Sensor Fusion Tutorial
Or how to avoid the infamous poor car <span>🚂🚃🚃💩🚃</span>

## What _is_ this sensor fusion thing?

This blog post is about sensor fusion. You might think you don&rsquo;t know what that means, but don&rsquo;t worry, you do. It&rsquo;s something you do all the time, as part of your daily life. In fact, a lot of it is done by your nervous system autonomously, so you might not even notice that it&rsquo;s there unless you look for it, or if something goes wrong.

Take, for example, riding a bicycle. If you know how to do it, you likely remember the frustrating days you spent getting on only to fall back down right away. In part, what made it so hard is that riding a bike requires several separate internal sensors to work in concert with one another, and at a high level of precision. Your brain must learn how to properly interpret and integrate visual cues with what your hands and feet perceive and with readings from your vestibular system (i.e., the &ldquo;accelerometers&rdquo; and &ldquo;gyroscopes&rdquo; in your inner ear). Perhaps going in a straight line is doable without visual cues, but I am willing to bet that taking proper turns while blindfolded is impossible for most people. That is because the visual cues add information about the state of the bicycle that none of the other sensors can provide such as what is the angle of the curve ahead, what angle am I currently turning at, and am I heading straight for a tree instead of the middle of the road? On the other hand, getting rid of the information from the vestibular system would be even more detrimental. The visual system, while very good at finding upcoming obstacles, is quite bad at determining small deviations from vertical, and it probably won&rsquo;t know you are falling until you are already halfway down.

Ok, so both sensors are needed for riding a bike successfully, but why fuse the information? Couldn&rsquo;t your eyes make sure that you&rsquo;re not about to hit an open door, while your inner ear makes sure you&rsquo;re staying upright? Of course, they could, but it turns out that integrating the two sources of data goes a long way toward eliminating noise from the measurement. Even though the eyes are very bad at determining equilibrium, they still provide some useful information. Your brain could throw this information away, or it could spend a tiny bit of extra energy and use it to significantly improve the accuracy of the inner ear sensor.

Engineers have been quick to catch onto this&mdash;at least soon after electronic sensors became a thing&mdash;and so have developed similar techniques to be used in control systems where the same variable can be measured in several different ways: the velocity of cars, the altitude of airplanes or the [attitude](https://en.wikipedia.org/wiki/Attitude_indicator) of space rockets. In the rest of the blog post I'll introduce some of these techniques while focusing on a relatively pedestrian end-goal: bettering my train commute.

## Sensing the number of people in a train car

Here at Datascope most of us take the CTA (Chicago Transit Authority) to work and back. My particular commute is 45 minutes each way on the [Red Line](https://en.wikipedia.org/wiki/Red_Line_(CTA)), which is the most crowded line in the city. As a result, it is sometimes hard to find a seat, and, on busy days, even to get on the train. Sometimes, like on baseball game days, this is unavoidable, since there are more people in the station than there is space on the train. On most days, however, this is merely a problem of distribution: some cars might be full, while others are only half full. Without waiting for the whole train to pass by and looking in every one of 20 or so cars, there is no way to know.

So, keeping this in mind, what could we do if we could measure or estimate the number of people in each car, in real time? For one, we could display a distribution of car occupancy, even before the train arrives. People could wait outside the emptiest car, and not try and crowd into an already crowded one. This would not just make the rides more comfortable, but also speed up boarding, which is a major factor of train delays, at least according to the [Chicago Transit Authority courtesy guide](http://www.transitchicago.com/courtesy/).

In addition, the occupancy estimate could solve a personal pet peeve of mine: [the poop car](http://www.redeyechicago.com/redeye-avoiding-poop-winter-cta-20141117-story.html). This is a Chicago phenomenon that happens in the winter: you think you scored a nice, relatively empty car, at a busy time too! It looks almost too good to be true, and it turns out it is. The car is filled with some stench or another, making it nearly impossible to breathe, so much so that, at the next stop almost all riders get off and switch to other, more crowded cars, while a new set of &ldquo;suckers&rdquo; walks into the trap. Broad City illustrated this phenomenon admirably:

<div style="text-align:center">
<img alt="Broad City smells a poo" src="broad-city.gif" style="width:100%">
</div>

Now, if we had a good estimate of the number of people in each car, we could easily see that one car is much emptier than the rest of the train, while the neighboring cars are fuller. For the Red Line in winter, that is a certain sign of either unbearable stench or some other, equally obnoxious problem.

## Sensor models

A simple and cheap way to get a (very rough) headcount in an enclosed space is to measure the level of carbon dioxide in the space. Since people breathe out more CO<sub>2</sub> than they breathe in, a higher concentration of CO<sub>2</sub> usually correlates with more people. How much, however, depends a lot on the space, (and maybe on other things, like light, time of day, or HVAC dynamics). There is no master function that says &ldquo;when your sensor reading 1000ppm of CO<sub>2</sub> there are 15 people in your room,&rdquo; and no such function can be calculated for a general case. This is why we need to build our own function, and we&rsquo;re going to do this by first running an experiment and using the data to build a sensor model. We install a cheap CO<sub>2</sub> sensor in each train car, and lump all sources of error into generic &ldquo;sensor noise&rdquo;. In our experiment we take readings of both the CO<sub>2</sub> level and the number of people present at that time in the train car, many times throughout a day. In code, it looks a little bit like this:

```python
class Sensor(object):
    ...
    def read(self, variable):
        """Given `variable` occupants, returns a sensor reading"""
        ...

    def fit(self, data):
        """Fits a sensor model to the experimental `data`"""
        ...

co2_sensor = Sensor(
    "CO_2", intersect=350, slope=15, sigma=10,
    round_level=500, proc_sigma=30, units="ppm")

for _ in range(datapoints):
    occupants = random.randrange(max_occupants + 1)
    sensor_data.append((occupants, co2_sensor.read(occupants)))

co2_sensor.fit(sensor_data)
```

Let&rsquo;s say we get a figure like this:

![CO<sub>2</sub> sensor model](experiment_plots/co_2.svg)

The above figure helps us build a sensor model for our particular car. A simple model fitting a line to the points in the figure tells us, for every given reading, how many people we expect to find in the room.[ref]Even though we use the model in a predictive fashion&mdash;given a reading, we look up the number of people that corresponds to that reading&mdash;we fit the line considering the number of occupants as the independent variable; doing it the other way around would generate a bad fit for our purposes[/ref]

We could stop here, report that number of people, and call it done. But we&rsquo;re no fools, we see there&rsquo;s a LOT of noise in the reading. And so, we also calculate the standard error for our fit, and get a rough estimate of how good our model is. In the above figures, you can see the uncertainty of our sensor model as a shaded gradient.

Let&rsquo;s talk about what that means for a minute. Say we check our sensor and get a reading of 1092ppm (a number I obtained by randomly mashing my keyboard). We see that this corresponds, on average, to there being approximately 46 people in the train car, but it&rsquo;s pretty much just as likely that there are 40 or 50 of them. The car could even be empty&mdash;that option is still within two standard deviations of the mean.[ref]If you think that this is a simplistic model, that&rsquo;s because it is. This is more or less the smallest building block of sensor fusion theory, and like any &ldquo;smallest block&rdquo; it has its problems. For example, you might have noticed that there&rsquo;s a non-zero probability there are negative occupants in the room. While it&rsquo;s true that we can leverage this fact to get a better predictor, allowing negative values for people is not strictly detrimental, as we can think of that as &ldquo;the reading was so low, it was even *more* unlikely there was anyone there&rdquo;[/ref]

![Sensor Fusion](reading_plots/1_co2.svg)

## Multiple readings between stations

Now that we have a sensor model, let&rsquo;s actually get some data! Let&rsquo;s start looking at what&rsquo;s happening in the car between two stations, something we can determine easily, either from an accelerometer on the car or from the great [train tracker app](http://www.transitchicago.com/traintracker/) CTA provides. The nice thing about the train between stations is that very few people change cars at that time (it is in fact illegal to do so). So, we can assume that the number of people in the car stays constant between measurements. We already have a sensor reading, so let&rsquo;s grab one more.

![Sensor Fusion](reading_plots/2_co2.svg)

This one is centered somewhere around 46 people and has, unsurprisingly, the same standard deviation as the first reading&mdash;this is a side effect of our model. Now comes the fun part: we merge the two readings into one probability curve:

![Sensor Fusion](reading_plots/3_co2.svg)

Mathematically, we multiply the two blue distributions and normalize the result as needed. Because our distributions are Gaussian this multiplication can be done analytically:
$$\mu = \frac{\sigma_{p}^2\mu_{up}+\sigma_{up}^2\mu_{p}}{\sigma_{up}^2+\sigma_{p}^2}, \quad \sigma=\frac{\sigma_{up}^2\sigma_{p}^2}{\sigma_{up}^2+\sigma_{p}^2}$$
In code, it looks something like this:

```python
class Gaussian(object):
    ...
    def bayesian_update(self, other):
        sigma_sum = self.sigma**2 + other.sigma**2
        self.mu = ((self.sigma**2) * other.mu + (other.sigma**2) * self.mu) / sigma_sum
        self.sigma = np.sqrt(((self.sigma * other.sigma)**2) / sigma_sum)
```

## Multiple sensors

We can apply the process in the previous section to multiple sensors that measure the same thing. For example, let&rsquo;s say we want to install a temperature sensor, whose measurements also (loosely) correlate with the number of people in the room. We install the new sensor:

```python
temp_sensor = Sensor(
  "Temperature", intersect=0, slope=0.25, sigma=5,
  round_level=10, proc_sigma=5, units="$^{\\circ}$C")
```

After doing the required experiments, the model for the temperature sensor looks like this :

![Temperature.png](experiment_plots/Temperature.svg)

Now, each measurement between the two stations could come from one sensor, or the other. Since we know how much to trust each sensor, we use the same method as before to fuse them together. The math does not change, since we&rsquo;re still multiplying Gaussians, and the same equations apply.

The first temperature measurement, for example, would fuse with the other two measurements like this:

![Sensor Fusion](reading_plots/4_co2.svg)

A 5 minute train ride might look like this, where the red line shows the true occupancy of the car:

<video src="5minutes.webm" controls="" style="width:100%">
    A 5 minute train ride between only two stations.
</video>

To make things easier, I made some custom classes to abstract away measurements, and used them as following:

```python
class Reading(Gaussian):

    def __init__(self, sensor, truth, timestamp=None):
        """Generate a reading from `sensor` at `timestamp`,
        given `truth` people"""
        ...

class Estimate(Gaussian):
    ...

    def add_reading(self, reading):
        self.reading_vector.append(reading)
        self.update(reading)

    def update(self, reading):
        ...
        self.bayesian_update(reading)

reading_1 = Reading(co2_sensor, 60)
reading_2 = Reading(temp_sensor, 60)

estimate = Estimate()
estimate.add_reading(reading_1)
estimate.add_reading(reading_2)

reading_3 = Reading(co2_sensor, 60)
estimate.add_reading(reading_3)
```

## Moving targets

Until now we assumed the number of people in the car stays constant. That&rsquo;s a decent assumption between two stations, but if we consider a whole route, this is certainly not true. Fortunately, the previous approach can be easily adapted to moving targets&mdash;and by this I mean &ldquo;changing number of people&rdquo;, not moving people between stations.

In order to estimate changing number of people we need to introduce the concept of time between measurements, and also build a model of human behavior between measurements[ref]in fact, we are already using a model for the human behavior between measurements: we model the number of people as non-changing[/ref]. Let&rsquo;s assume that, at every station, a random number of people enter or leave, except at the last station, when everyone leaves the train. In mathematical terms this means that, every time we stop at a station, the error on our estimate of the number of people in the car will have to widen, and since we don&rsquo;t know how much we&rsquo;ll just set its sigma to infinity. In code, this is done by either creating a brand new `Estimate` or setting its `sigma` to `None`:
```python
estimate=Estimate()
# or
estimate.sigma = None
```
    However, as the train is moving between the stations we will be getting measurements again, and our estimation will narrow. Over several stations our estimate will look something like this:

<video src="30minutes.webm" controls="" style="width:100%">
    A 30 minute train ride between several stations.
</video>

In the end we get a filtered version of the signal, translated from CO<sub>2</sub> and temperature levels to number of people in the room. We also get a confidence interval at every point in time. Perhaps, more importantly, we obtained these estimates in *real time*. What this means is that we didn&rsquo;t have to wait until the train ride was over, or until we got to the next station in order to average all the data and get a good estimate. Instead, we included every single measurement as it came in, and we could present the people waiting at the next station a constantly updating estimate of how many people were in each car.

## Better models

What we did in the previous sections is a well know, and quite brilliant method, called *Kalman filtering*. It&rsquo;s one of the technological advances that helped Apollo 11 get to the moon, and variants of it are used nowadays on pretty much any non-trivial control system. An old joke says that every time the late Rudolf Kálmán got on a plane he would state his name and ask to be shown &ldquo;his filter&rdquo;. A very similar approach, called *Kalman smoothing* can be taken if we do not need real time results, but instead can use the full data at once to determine the most likely outcome. In our case (due to the fact that we assume no movement between cars between stations ) a Kalman smoother would have resulted in the estimate right before each station being considered as the estimate for that whole leg of the trip. Because of its overall much better estimate, smoothing is usually preferred to filtering when real-time results are not necessary. The downside would be that we wouldn&rsquo;t be able to tell which car is the smelly car until the end of the line, and then it&rsquo;s no better than a quick in-person smell test of the whole train.

Improvements on this approach are numerous, and I&rsquo;m sure some of you are already thinking about what they might be. I&rsquo;ll list only a few:

  - non-linear sensor models: we fit a polynomial or exponential curve to the experimental data
  - non-constant sensor noise model: the standard deviation is different at different sensor readings, not constant across the range of readings
  - time-varying sensor model: the way the sensors behave might change with time of day, or other factors (e.g. air conditioning)
  - more complex human dynamics: the way people enter and leave a car is a lot more complicated than we assumed; we could, for example, take into account that people are more likely to enter or leave during rush hour; people might prefer to enter into emptier cars; or they might be actively leaving the poop car
  - some other stuff I haven&rsquo;t thought of, but you probably have: add it to the comments!

## See also

I purposefully kept this post light, especially in terms of math. Just a quick look at the wikipedia page for Kalman filtering is enough to turn most people off the whole subject. I wanted to make sure that you first and foremost follow the concepts presented here intuitively, rather than mathematically. However, I bet there are some of you who want more &ldquo;meat&rdquo; than what I&rsquo;ve given here. I hear you. Consider this an appetizer. I&rsquo;ll let you do your own cooking for the main meal, but I&rsquo;ll give a list of potential ingredients:

  - [the github repo](https://github.com/datascopeanalytics/sensor_fusion) with the full code backing this post
  - [a succint technical version of this presentation, for those familiar with complex statistics notation](http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/kalman.pdf)
  - the Wikipedia page for [Kalman filtering](https://en.wikipedia.org/wiki/Kalman_filter)
  - [pykalman](https://pykalman.github.io/), a very simple python library I have used in the past
  - Steve LaValle&rsquo;s relatively accessible [blog post](https://developer3.oculus.com/blog/sensor-fusion-keeping-it-simple/) on how they do sensor fusion for the Oculus while avoiding Kalman Filtering
  - a very nice and simple explanation of [particle filtering](https://www.youtube.com/watch?v=aUkBa1zMKv4), which replaces assumptions of Gaussian distributions with hard work (on the part of the computer)
