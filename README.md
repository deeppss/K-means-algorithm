# K-means clustering algorithm

Drivers for Uber, Lyft, and other Transportation Network Companies need
to find the most profitable areas to drive and the most profitable times to
drive. Cab drivers would like to know what spots in the city would
increase their chances of getting a ride request and what time of the day
would have the highest traffic to maximize their earnings.


We can use the K-means clustering algorithm to pinpoint cluster centers
within the traffic data. Companies would be able to improve their annual
revenue and can expand their business by choosing the right location.
First, we need to collect data from a large cab rides dataset (from Kaggle)
and clean it, so that we can perform analysis on it. Then, we would analyze
the pickup and drop-off points that occurred over a period of time within a
given area. From the dataset, we now implement the K-means clustering
technique.

First, we need to assign each pickup point to a cluster. “K” in K-means
represents the number of clusters. We can determine the optimal number of
clusters into which the data may be clustered using the elbow method.
Once we assign each pick-up data point to a cluster, we need to rank the
clusters based on pickup volume. Then, we have to calculate the distance
between each data point and the cluster using the Euclidean distance
function. We need to group each pickup point based on how close it is to
the cluster. Now we calculate the mean of each cluster. We keep
measuring the distance between the centroid and data point, then cluster
them using the mean values until there is no change in clustering between
two consecutive iterations.

Finally, we represent these groups of clusters in graphical format by
importing libraries. These graphs help companies identify pickup areas
with high traffic.
