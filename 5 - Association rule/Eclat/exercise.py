import pandas as pd
from apyori import apriori
from tabulate import tabulate

# Data preprocessing
dataset = pd.read_csv(r"5 - Association rule\datasources\market_basket_optimisation_data.csv",header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])


#print(transactions)

# Train the model
rules = apriori(
    transactions=transactions,
    min_support = 0.003,  # 3 * 7 / 7501 = 003 ( Product that appear at least 3 times per day per week)
    min_confidence = 0.2,
    min_lift = 3, # Use minimum 3
    min_length = 2, # Buy one get one free case study
    max_length = 2, # Buy one get one free case study
)

# Display the first results
results = list(rules)

# Putting the results well organized
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports)
    )

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ["Product 1", "Product 2", "Support"])


print(tabulate(resultsinDataFrame.nlargest(n=10, columns="Support"), headers='keys', tablefmt='pretty'))