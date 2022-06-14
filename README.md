# Project_05_IncomePrediction
### <b>ABOUT</b>
A globally-renowned bank plans to target new potential customers, particularly young individuals, to open their first bank account. However, they primarily want to target prospects most likely to have high incomes later in life.

This project is tasked with building a model to determine a person's potential income. The data provided only consists of the parents' income (the company will be targeting the children of their current clients) and their country. With so little data available, it seems like quite a challenge!

Thus, we will suggest a linear regression with 3 variables:

- Parental income
- Average income of the prospect's country
- Gini index calculated on the income of the inhabitants per country
This project deals only with the construction and interpretation of the model and not until the forecasting phase.

### <b>CONDITIONS</b>
For this project, it is required to perform a descriptive statistical analysis in Python (with graphical representations). It will also be necessary to apply ANOVA or linear regression type modeling.


### <b>SKILLS ASSESSED</b>
- Master the basics of inferential statistics
- Master the basics of probability
- Model data

### <b>DATA</b>
This file contains the 2008 [World Income Distribution](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/data-projet7.csv) data. This database mainly comprises studies carried out at the national level for many countries and includes the income distributions of the populations concerned.

Addition data used in this project is the Gini indices estimated by the World Bank, available through this [link](http://data.worldbank.org/indicator/SI.POV.GINI). You are also free to find other sources or recalculate the Gini indices from the World Income Distribution if you want to recreate this project.

You will also need to retrieve the number of populations for each country in the database. You can retrieve them from [FAO](https://www.fao.org/faostat/en/#data/OA)'s website or [The World Bank](https://data.worldbank.org/indicator/SP.POP.TOTL?view=chart).

# <b>MISSIONS</b>
### Mission n°1
Summarize the data:
- number of years in the data
- number of countries in the data
- population covered by the analysis (in terms of percentage of the world population)

The World Income Distribution data contains quantiles of the income distribution for each country. Describe the data:
- What kind of quantiles are used (quartiles, deciles, etc.)?
- Do you think sampling a population using quantiles is a suitable method? Why?

> *Each quantile represents a class of income and is the average income for each class or group of individuals.*

The unit used in the World Income Distribution income variable is $PPP. This unit is calculated by the World Bank using the Eltöte-Köves-Szulc method.
- What does this unit correspond to, and why is it relevant to comparing countries?

### Mission n°2
- Demonstrate the diversity of countries in terms of income distribution using a graph. This will represent the average income (y-axis, on a logarithmic scale) of each income class (x-axis) for 5 to 10 (*chosen*) countries
Represent the Lorenz curve for each chosen country
- Illustrate the evolution of the Gini index over the years
- Rank countries by Gini index
- Average of 5 countries with the highest Gini index
- Average of 5 countries with the lowest Gini index
- France's ranking in the Gini index

### Mission n°3
At this point, we have 2 out of 3 explanatory variables needed:
- $m_{j}$ = the average income of country: (data_m2)
- $G_{j}$ = the Gini index of country: (data)

We need the parents' income group $c_{i}$ for each child $i$

> We'll assume that each child $i$ is associated with a unique group $c_{i, parent}$; regardless of the number of parents.

Therefore, we will simulate this information using a coefficient ${\rho}_{j}$ (specific to each country $j$), measuring a correlation between the income of the child $i$ and his parents'. This is the *elasticity coefficient*; it measures *intergenerational income mobility*.

> For more information on the elasticity coefficient calculation, you can consult this [document](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/2011-measuring-intergenerational-income-mobility-art.pdf) (equation 1 on page 8 in particular). This coefficient is determined by a simple linear regression in which the logarithm of the child's income $Y_{ child}$ is a function of the parents' income $Y_{parent}$ logarithm:
>
> $ln(Y_{child}) = \alpha + \rho_{j} ln(Y_{parent}) + \epsilon$

Two possibilities are available to obtain the elasticity coefficient :
- Based on the coefficients in the [GDIM dataset](https://s3.eu-west-1.amazonaws.com/course.oc-static.com/projects/DANV1_P7/GDIMMay2018+(1).csv) given by The World Bank. The elasticity coefficient for each country is provided under the IGE income variable.
- Based on the estimations from multiple studies (*gathered from different regions of the world*) in the [elasticity.txt](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/parcours-data-analyst/projet_7.zip) file. Please note that this source can be outdated.

> *It is also possible to combine these two approaches.*

For each country, we will use a random generation of the parents' income group solely from these two pieces of information:
- ${\rho}_{j}$
- income group of the child $c_{i, child}$

Make sure to refer to the child's income group (between 1 and 100 if you use 100 quantiles) rather than their PPP income. Similarly, we are not trying to generate the parents' income but the parents' income group $c_{i, parent}$

Here is the generation protocol for a given country $j$, which is based on the equation given above:

> A sample code for performing operations 1 to 6 is at the bottom. You are free to use it. In particular, `the proba_cond` function will give you the probabilities $P(c_{i, parent} | c_{i, child}, j)$.

1. According to Gaussian's normal law, we will generate a large number $n$ of realizations of the $ln(Y_{child})$ variable. The choice of the mean and the standard deviation will not affect the final result. $n$ must be greater than 1000 times the number of quantiles.
2. Generate $n$ realizations of the error term $\epsilon$ according to a normal distribution with a mean=0 and a standard deviation=1.
3. For a given value of$\rho_{j}$ (e.i., 0.9), compute $Y_{child} = e^{\alpha+\rho_j ln(Y_{parent})+\epsilon}$ . The choice of $alpha$ does not affect the final result and can be deleted. At this stage, $Y_{child}$ contains values whose order of magnitude does not reflect reality, but this has no influence later.
4. For each of the $n$ individuals generated, the income group of the child $c_{i, child}$ as well as the income group of its parents $c_{i, parent}$ are calculated from $Y_{child}$ and $Y_{parent}$
5. The conditional distribution of $c_{i, parents}$ for each $c_{i, child}$ is estimated.
- For example, if we observe 6 individuals having both $c_{i, child}=5$ and $c_{i, child}=8$, and that 200 individuals out of 20 000 have $c_{i, child}=5 $, then the probability of having $c_{i, parent}=8$ knowing $c_{i, child}=5$ and knowing $\rho_{j}=0.9$ will be estimated at 6/200.
- We note this probability like this: $P(c_{i, parent}=8 | c_{i, child}=5, \rho_{j}=0.9) = 0.03$.
- If the population is divided into $c$ income groups, we should then have $c^2$ estimates for each country's conditional probabilities.
6. (*Optional*) To check the consistency of the code, we can create a graph representing these conditional distributions. Here are 2 examples for a population segmented into 10 classes, for 2 values of $\rho_{j}$:
- one reflecting high mobility (0.1)
- one with low mobility (0.9)

<img width="404" alt="image" src="https://user-images.githubusercontent.com/93476912/173585576-4ccdcf3d-7126-45f2-9568-3b15c81583ed.png">

### Mission n°4
For mission 4, we will seek to explain the income of individuals according to several explanatory variables:
- Country of the individual
- Country's Gini index
- Income group of the parents
- etc.

1. We will apply ANOVA to our data, using only the individual's country as an explanatory variable. We will analyze the model's performance.

> *For each of the following regressions, we will test 2 versions:*
> - one by using the average income of the country and the income of parents & children in logarithm (ln)
> - the other by leaving them as they are. We will choose the most efficient version to answer the questions.

2. We will apply a linear regression to our data, including as explanatory variables only the average income of the individual's country and the Gini index of the individual's country. What is the percentage of variance explained by your model?

Give the total variance decomposition based on the model explained by:
- country of birth (i.e., average income and Gini index)
- other factors not considered in the model (effort, luck, etc.)

3. We will improve the previous model by including parents' income groups this time. What is the percentage of variance explained by this new model?

> *By observing the regression coefficient associated with the Gini index, can we say that living in an unequal country favors more people than it disadvantages?*

4. Give the total variance decomposition based on the previous model explained by:
- country of birth and parental income
- other factors not considered in the model (effort, luck, etc.)

# <b>ANNEXE : code</b>
Here is the code mentioned in mission 3, up to you to use it or not:
```python
import scipy.stats as st
import pandas as pd
import numpy as np
from collections import Counter

def generate_incomes(n, pj):
    # On génère les revenus des parents (exprimés en logs) selon une loi normale.
    # La moyenne et variance n'ont aucune incidence sur le résultat final (ie. sur le caclul de la classe de revenu)
    ln_y_parent = st.norm(0,1).rvs(size=n)
    # Génération d'une réalisation du terme d'erreur epsilon
    residues = st.norm(0,1).rvs(size=n)
    return np.exp(pj*ln_y_parent + residues), np.exp(ln_y_parent)
    
def quantiles(l, nb_quantiles):
    size = len(l)
    l_sorted = l.copy()
    l_sorted = l_sorted.sort_values()
    quantiles = np.round(np.arange(1, nb_quantiles+1, nb_quantiles/size) -0.5 +1./size)
    q_dict = {a:int(b) for a,b in zip(l_sorted,quantiles)}
    return pd.Series([q_dict[e] for e in l])

def compute_quantiles(y_child, y_parents, nb_quantiles):
    y_child = pd.Series(y_child)
    y_parents = pd.Series(y_parents)
    c_i_child = quantiles(y_child, nb_quantiles)
    c_i_parent = quantiles(y_parents, nb_quantiles)
    sample = pd.concat([y_child, y_parents, c_i_child, c_i_parent], axis=1)
    sample.columns = ["y_child", "y_parents", "c_i_child","c_i_parent"]
    return sample

def distribution(counts, nb_quantiles):
    distrib = []
    total = counts["counts"].sum()
    
    if total == 0 :
        return [0] * nb_quantiles
    
    for q_p in range(1, nb_quantiles+1):
        subset = counts[counts.c_i_parent == q_p]
        if len(subset):
            nb = subset["counts"].values[0]
            distrib += [nb / total]
        else:
            distrib += [0]
    return distrib   

def conditional_distributions(sample, nb_quantiles):
    counts = sample.groupby(["c_i_child","c_i_parent"]).apply(len)
    counts = counts.reset_index()
    counts.columns = ["c_i_child","c_i_parent","counts"]
    
    mat = []
    for child_quantile in np.arange(nb_quantiles)+1:
        subset = counts[counts.c_i_child == child_quantile]
        mat += [distribution(subset, nb_quantiles)]
    return np.array(mat) 

def plot_conditional_distributions(p, cd, nb_quantiles):
    plt.figure()
    
    # La ligne suivante sert à afficher un graphique en "stack bars", sur ce modèle : https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html
    cumul = np.array([0] * nb_quantiles)
    
    for i, child_quantile in enumerate(cd):
        plt.bar(np.arange(nb_quantiles)+1, child_quantile, bottom=cumul, width=0.95, label = str(i+1) +"e")
        cumul = cumul + np.array(child_quantile)

    plt.axis([.5, nb_quantiles*1.3 ,0 ,1])
    plt.title("p=" + str(p))
    plt.legend()
    plt.xlabel("quantile parents")
    plt.ylabel("probabilité du quantile enfant")
    plt.show()
    
def proba_cond(c_i_parent, c_i_child, mat):
    return mat[c_i_child, c_i_parent]

pj = 0.9                 # coefficient d'élasticité du pays j
nb_quantiles = 100       # nombre de quantiles (nombre de classes de revenu)
n  = 1000*nb_quantiles   # taille de l'échantillon

y_child, y_parents = generate_incomes(n, pj)
sample = compute_quantiles(y_child, y_parents, nb_quantiles)
cd = conditional_distributions(sample, nb_quantiles)
#plot_conditional_distributions(pj, cd, nb_quantiles) # Cette instruction prendra du temps si nb_quantiles > 10
print(cd)

c_i_child = 5 
c_i_parent = 8
p = proba_cond(c_i_parent, c_i_child, cd)
print("\nP(c_i_parent = {} | c_i_child = {}, pj = {}) = {}".format(c_i_parent, c_i_child, pj, p))
```
