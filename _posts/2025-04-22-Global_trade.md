---
layout: post
title:  "Global Trade Networks through Rice and Wheat: A Data-Driven View of Global Flows"
---


<div style="text-align: center;">
    This project is from my class. Feel free to look around.
</div>

<div style="font-size: 36px; text-align: center;">
    Source
</div>

<p style="text-align: center;">
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2021) Deng et al.pdf" target="_blank">(2021) Deng et al.pdf.</a>
</p>

<div style="font-size: 36px; text-align: center;">
  <b>1. Summary of the reading</b>
</div>

<div style="font-size: 18px; text-align: center;">
  <b> • Research Question & Research Gap</b>
</div>

<div style="text-align: center;">
- Research question: How does virtual water flow between countries, and how can we use social network analysis (SNA) to understand the global structure of this flow?
</div>

<div style="text-align: center;">
- Research Gap: Most past research focused only on agricultural products and didn’t explore the whole-industry level or apply SNA to network characteristics.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Method</b>
</div>

<div style="text-align: center;">
What is used: MRIO model (EORA26) + SNA (Density, Asymmetry, In/Out-Degree)
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Results + Interpretation</b>
</div>

<div style="text-align: center;">
- Virtual water trade increased over time, with rising density and imbalance.
</div>

<div style="text-align: center;">
- China led in exports (Out-Degree), USA in imports.
</div>

<div style="text-align: center;">
- Agriculture sector had the densest and most asymmetric water trade.
</div>

<div style="text-align: center;">
- Virtual water trade helps address water scarcity but needs logistics investment and sectoral balancing
</div>

<br>

<div style="font-size: 36px; text-align: center;">
  <b>2. Data introduction</b>
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Data sources</b>
</div>

<div style="text-align: center;">
Data I used is from the Food and Agriculture Organization (FAO). 
</div>

<div style="text-align: center;">
<a href="https://www.fao.org/faostat/en/#data" target="_blank">The site where the data from.</a>
</div>

<div style="text-align: center;">
I used the data which is in the section 'Trade' and in 'Detailed trade matrix' and chose Rice and Wheat and import quantity and export quantity. And both sexes population data and macro indicators with Value US$ per capita.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Major characteristics of the data</b>
</div>


![Original Dataset](/assets/images/original_dataset.png)


<br>

<div style="text-align: center;">
This is the original dataset from FAO. The dataset contains over 50 million rows and 9 columns, making it a high-resolution, large-scale trade matrix. This records both import and export flows, with each flow represented as a directed entry from one country to another.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Data pre-processing</b>
</div>


<div style="text-align: center;">
Before I do the analysis for wheat and rice I had to clean the dataset, so I can use the data that I need it.
</div>

<div style="text-align: center;">
- So I separate the data from original dataset to wheat export and import.
</div>

```python

df2 = df[(df["Value"]>0) & (df["Element"] == "Export quantity") & (df["Item"] == "Wheat")]

```

```python
df3 = df[(df["Value"]>0) & (df["Element"] == "Import quantity") & (df["Item"] == "Wheat")]
```

<div>
- And merged them together with Value_x as export quantity and Value_y as import quantity. And saved the bigger one into Final Value.
</div>

```python
wheat = pd.merge(
    df2,
    df3,
    on=["Reporter Country Code", "Reporter Countries", "Partner Country Code", "Partner Countries", "Year", "Item"],
    suffixes=("_x", "_y")
)

wheat = wheat[[
    "Reporter Country Code",
    "Reporter Countries",
    "Partner Country Code",
    "Partner Countries",
    "Year",
    "Value_x",  # Export
    "Value_y"   # Import
]]

wheat["Item"] = "Wheat"

wheat["Final Value"] = wheat[["Value_x", "Value_y"]].max(axis=1)
```



<div style="text-align: center;">
- And same process on rice.
</div>

```python
df4 = df[(df["Value"]>0) & (df["Element"] == "Export quantity") & (df["Item"] == "Rice")]
```

```python
df5 = df[(df["Value"]>0) & (df["Element"] == "Import quantity") & (df["Item"] == "Rice")]
```

```python
rice = pd.merge(
    df4,
    df5,
    on=["Reporter Country Code", "Reporter Countries", "Partner Country Code", "Partner Countries", "Year", "Item"],
    suffixes=("_x", "_y")
)

rice = rice[[
    "Reporter Country Code",
    "Reporter Countries",
    "Partner Country Code",
    "Partner Countries",
    "Year",
    "Value_x",  # Export
    "Value_y",  # Import
]]

rice["Item"] = "Rice"

rice["Final Value"] = rice[["Value_x", "Value_y"]].max(axis=1)
```

<div style="text-align: center;">
- And prepared for multiplex network construction by integrating node-to-node connection data.
</div>

```python
link = pd.concat([rice, wheat], ignore_index=True)
```

<div style="text-align: center;">
- And I unified column names to suitable format for network analysis + leave only necessary columns.
</div>

```python
link = link.rename(columns={
    "Reporter Country Code": "O_code",
    "Reporter Countries": "O_name",
    "Partner Country Code": "D_code",
    "Partner Countries": "D_name",
    "Year": "year",
    "Item": "item",
    "Final Value": "weight"
})[["O_code", "O_name", "D_code", "D_name", "year", "item", "weight"]]
```


![Original_Link](/assets/images/original_link.png)


<br>

<div style="text-align: center;">
And I loaded the population and macro indicators dataset and modified them to have information that I need.
</div>

<div style="text-align: center;">
- Population data
</div>

```python
pop = pd.read_csv('Population_E_All_Data_(Normalized).csv')
```

```python
pop = pop[["Area Code", "Area", "Item", "Element", "Year", "Unit", "Value"]]
```

```python
pop1 = pop[(pop["Value"]>0) & (pop["Element"] == "Total Population - Both sexes") & (pop["Year"] <= 2023)]
pop1 = pop1[(pop1["Area Code"]<5000)]
```

<div style="text-align: center;">
- Macro indicators data
</div>

```python
mac = pd.read_csv("Macro-Statistics_Key_Indicators_E_All_Data_(Normalized).csv", encoding="cp1252")
```

```python
mac = mac[["Area Code", "Area", "Item", "Element", "Year", "Unit", "Value"]]
```

```python
mac1 = mac[(mac["Value"]>0) & (mac["Element"] == "Value US$ per capita")]
mac1 = mac1[(mac1["Area Code"]<5000)]
```

<div style="text-align: center;">
And I merged them to make a node dataset
</div>

```python
pop1 = pop1.rename(columns={
    "Area": "country",
    "Area Code": "country_code"
})

mac1 = mac1.rename(columns={
    "Area": "country",
    "Area Code": "country_code"
})

pop1["country_code"] = pop1["country_code"].astype(str).str.zfill(3)
mac1["country_code"] = mac1["country_code"].astype(str).str.zfill(3)
pop1["country"] = pop1["country"].str.strip()
mac1["country"] = mac1["country"].str.strip()

node = pd.merge(
    pop1, mac1,
    on=["country", "country_code", "Year"],
    how="inner"
)
```


![Node](/assets/images/node.png)


<br>

<div style="font-size: 36px; text-align: center;">
  <b>3. Analysis</b>
</div>

<div style="text-align: center;">
To analyze the datasets, I first looked at the desities of rice and wheat through 1986 to 2023.
</div>


![Density_of_Rice_and_Wheat](/assets/images/density_of_rice_and_wheat.png)


<div style="font-size: 18px; text-align: center;">
<b>• Graph explanation: </b>
</div>


<div style="text-align: center;">
- Y-axis (Density): Represents the ratio of actual trade connections to all possible connections in the network. A higher value means more countries are actively trading with each other.
</div>

<div style="text-align: center;">
- X-axis (Year): Spans from 1985 to 2022.
</div>

<br>

<div style="text-align: center;">
- A noticeable feature shows in between 1988 and 1922.
</div>

<div style="text-align: center;">
- This suggests a period of unusually high connectivity in rice trade, possibly due to policy shifts or global trade agreements.
</div>


<div style="text-align: center;">
- After those years, the rice trade network became significantly less dense and settled around a low, stable value, indicating fewer or more selective trade links.
</div>

<div style="text-align: center;">
- Since around 2010, the densities of rice and wheat networks have become nearly equal, indicating a possible convergence in the global trade structures for these two staple crops.
</div>

<div style="text-align: center;">
- The rice trade network was very connected at some times, but then broke into parts. The wheat network stayed stable and spread out. In recent years, both networks look similar, which shows they may be trading in the same way.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
<b>• Out-degree and In-degree Distributions </b>
</div>

<div style="text-align: center;">
- I calculated weighted in-degrees and out-degrees for all countries to identify dominant importers/exporters for 2023 year. And checked the top five exporting and importing countries 
</div>

![indegree](/assets/images/indegree.png)

<div style="text-align: center;">
- So, In-degree means how many other countries a country imports from.
</div>

<div style="text-align: center;">
- France has the highest.
</div>

<div style="text-align: center;">
- Germany and Italy are also strong importers.
</div>


![outdegree](/assets/images/outdegree.png)


<div style="text-align: center;">
- Out-degree means how many other countries a country exports to.
</div>

<div style="text-align: center;">
- France exports to the most countries and plays a major role in global grain trade. And showing it is active in both export and import.
</div>

<div style="text-align: center;">
- Germany and Italy are also strong importers.
</div>

<div style="text-align: center;">
- Many European countries play a key role in connecting the global market.
</div>

<div style="text-align: center;">
- Some countries are active in both exporting and importing, showing they have multiple roles in international trade.
</div>

<br>

<div style="text-align: center;">
• And I looked at these 6 countries to see the patterns over 1986 to 2023. 
</div>


![indegreeover](/assets/images/indegreeover.png)


<div style="text-align: center;">
- France has always been the top importer with very high in-degree over the years.
</div>

<div style="text-align: center;">
- Germany and Italy also have steady in-degree values, showing that they consistently receive many connections in the trade network.
</div>

<div style="text-align: center;">
- Germany and Italy consistently receive many connections in the trade network.
</div>

<div style="text-align: center;">
- Overall, the top 3 countries remain stable, while the rest show smaller or more unstable import trends.
</div>


![outdegreeover](/assets/images/outdegreeover.png)


<div style="text-align: center;">
- France is again the highest.
</div>

<div style="text-align: center;">
- Germany, Italy, and Canada follow, with similar upward trends over time.
</div>

<div style="text-align: center;">
- Morocco also increases gradually, but its level remains relatively low.
</div>

<div style="text-align: center;">
- Overall, France stays the top country in both importing and exporting. Germany and Italy also play big roles. Kazakhstan and Morocco started small but are growing in the trade network.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
<b>• Network Visualization: </b>
</div>

<div style="text-align: center;">
- Using force-directed, I visualized core trade networks, incorporating node size (betweenness centrality) and node color (modularity). This allowed identification of central actors and clustered communities.
</div>

![network](/assets/images/network.png)


<div style="text-align: center;">
- The size of each node shows its betweenness centrality – how important it is as a connector between other countries.
</div>

<div style="text-align: center;">
- Colors represent different modularity communities, meaning countries grouped by how closely they are connected within clusters.
</div>

<div style="text-align: center;">
- France, Germany, and Italy are central and large, meaning they play major roles in connecting global trade flows.
</div>

<div style="text-align: center;">
- Big and central countries like France and Germany connect many others. Countries with the same color often trade more with each other. Some countries are not very connected and are at the edges.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
- Rice trade map
</div>





<iframe src="/assets/rice_trade_map.html" width="100%" height="700px" frameborder="0"></iframe>

<div style="text-align: center;">
- This map shows the global rice trade network using nodes and curved lines
</div>

<div style="text-align: center;">
- Red circles represent countries.
</div>

<div style="text-align: center;">
- The larger the circle, the more rice that country exports.
</div>

<div style="text-align: center;">
- Red curved lines indicate the trade routes.
</div>

<div style="text-align: center;">
- Thicker lines show higher trade volume.
</div>

<div style="text-align: center;">
- India, Thailand, and Vietnam are major exporters, as shown by their large red circles. They dominate rice exports globally.
</div>

<div style="text-align: center;">
- Africa and parts of the Middle East receive many trade links, suggesting they are key importers of rice.
</div>


<br>

<div style="font-size: 18px; text-align: center;">
- Wheat trade map
</div>

<iframe src="/assets/wheat_trade_map.html" width="100%" height="700px" frameborder="0"></iframe>

<div style="text-align: center;">
- This map shows the international wheat trade network using nodes and curved edges
</div>

<div style="text-align: center;">
- Blue circles represent countries.
</div>

<div style="text-align: center;">
- The bigger the circle, the more wheat that country exports
</div>

<div style="text-align: center;">
- Red curved lines show wheat trade flows between countries.
</div>

<div style="text-align: center;">
- The thicker the line, the larger the trade volume.
</div>

<div style="text-align: center;">
- Overall, This map is showing France, Germany, and Canada as major exporters. Red trade flows show how wheat moves mainly from Europe and North America to Africa and Asia, revealing the structure and direction of global wheat distribution.
</div>

