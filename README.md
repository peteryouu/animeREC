# animeREC

The above project was done using sklearn's NearestNeighbor and PySpark. To expedite the EDA process, I used the AutoViz library to plot the predictors.

The shortcomings of using a K-nearest neighbor approach is that it recommended only popular shows, which is to be expected, as kNN measures the distance (euclidean/cosine) of nearby instances. It then picks the most popular class amongst its neighbors. The problem therein lies with itself, as new shows and less popular shows would not be represented. From a more technical standpoint there's two major shortcomings with using the kNN approach: 

*  kNN's major shortcoming is popularity bias, where our system will only recommend shows that have already been widely seen and rated. This may not be what our user is lookign for, as one who's seen Naruto, for example, would have also seen the recommendations (One Piece, Bleach, etc.,).

* The second major shortcoming would go in hand with the first point, inwhich our system does not recommend newer shows, due to no interactions in our model. This is known as the cold-start problem.

From Netflix's famous recommender system challenge, the approach was to go with a hybrid-based approach, one that uses matrix factorization to elucidate the latent features underlying the interactions between users and shows. One of the approaches with finding such latent features was with singular value decomposition. The math is beyond the scope of this personal project, however given our dataset of approximately 57 million rows of ratings, such decomposition lacks scalability. Instead, using Spark, we use its alternating least squares function, which provides a good tradeoff between matrix factorization and scalability.

From the notebook, we see that despite having tuned our model, its recommendations after given Naruto as a test input were all shows I've never watched. These shows are all low ranking on MyAnimeList as well, which suggests that more work is required in tuning the parameters, and that instead of being a sole product, we could combine the kNN approach with this ALS-based approach to create a hybrid recommendation system to better suit the needs of the potential users. This way, I'll get users that get recommended shows that address the major shortcomings with kNN, as well as have popular shows that users are more likely to watch.


The dataset was retrieved from https://www.kaggle.com/hernan4444/anime-recommendation-database-2020
