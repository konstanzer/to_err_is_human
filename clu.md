
## Zillow: What is driving the **errors** in the Zestimates?

Continue to use the 2017 properties and predictions data for single unit properties.

Audience: data science team.

Presentation: **notebook** demo w.r.t. drivers of the log error.

Specification: repository with the following contents:

* final notebook w/ markdown documentation and clean code.
* README that explains what the project is, how to reproduce you work, and your notes from project planning
* Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.

**Acquisition**: Data is collected from the codeup cloud database 
**Prep**: Column data types are appropriate for the data they contain, Missing values are investigated and handled, Outliers are investigated and handled
**Exploration**: the interaction between independent variables and the target variable is explored using visualization and statistical testing. Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are useful. At least 3 combinations of features for clustering should be tried.
**Modeling**: At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.

Guidance: Acquisition here can take some time. You should probably build out caching in your python scripts to store the data locally as a csv in order to speed up future data acquisition. Create sections indicated with markdown headings in your final notebook the same way you would create seperate slides for a presentation. For your MVP, do the easiest thing at each stage to move forward. Model on scaled data, explore on unscaled.

Clustering could be useful in several ways on this project:
* Do clusters produce an interesting insight, takeaway, or visualization that can be shared and communicated?
* With a small number of clusters, clusters could be one-hot encoded and used as a feature for modeling.
* Different models can be created for different clusters (while conceptually simple, this involves a much more complicated python implementation, so you should probably treat this idea as a bonus)

Sometimes your conclusion is that there is no effect or no significant difference. This is a valid conclusion in and of itself.
