---
layout: post
title:  "Global Trade Networks through Rice and Wheat: A Data-Driven View of Global Flows"
---


<div style="text-align: center;">
    This project is from my class. Feel free to look around.
</div>

<div style="font-size: 36px;">
    Source
</div>
<p>
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2021) Deng et al.pdf" target="_blank">(2021) Deng et al.pdf.</a>
</p>

<div style="font-size: 36px;">
  <b>1. Summary of the reading</b>
</div>

<div style="font-size: 18px;">
  <b> • Research Question & Research Gap</b>
</div>

<div>
- Research question: How does virtual water flow between countries, and how can we use social network analysis (SNA) to understand the global structure of this flow?
</div>

<div>
- Research Gap: Most past research focused only on agricultural products and didn’t explore the whole-industry level or apply SNA to network characteristics.
</div>

<br>

<div style="font-size: 18px;">
  <b> • Method</b>
</div>

<div>
What is used: MRIO model (EORA26) + SNA (Density, Asymmetry, In/Out-Degree)
</div>

<br>

<div style="font-size: 18px;">
  <b> • Results + Interpretation</b>
</div>

<div>
- Virtual water trade increased over time, with rising density and imbalance.
</div>

<div>
- China led in exports (Out-Degree), USA in imports.
</div>

<div>
- Agriculture sector had the densest and most asymmetric water trade.
</div>

<div>
- Virtual water trade helps address water scarcity but needs logistics investment and sectoral balancing
</div>

<br>

<div style="font-size: 36px;">
  <b>2. Data introduction</b>
</div>

<br>

<div style="font-size: 18px;">
  <b> • Data sources</b>
</div>

<div>
Data I used is from the Food and Agriculture Organization (FAO). 
</div>

<div>
<a href="https://www.fao.org/faostat/en/#data" target="_blank">The site where the data from.</a>
</div>

<div>
I used the data which is in the section 'Trade' and in 'Detailed trade matrix' and chose Rice and Wheat and import quantity and export quantity. And both sexes population data and macro indicators with Value US$ per capita.
</div>

<br>

<div style="font-size: 18px;">
  <b> • Major characteristics of the data</b>
</div>


![Original Dataset](/assets/images/original%20dataset.png)

<br>

<div>
This is the original dataset from FAO. The dataset contains over 50 million rows and 9 columns, making it a high-resolution, large-scale trade matrix. This records both import and export flows, with each flow represented as a directed entry from one country to another.
</div>

<br>

<div style="font-size: 18px;">
  <b> • Data pre-processing</b>
</div>


<div>
Before I do the analysis for wheat and rice I had to clean the dataset, so I can use the data that I need it.
</div>

<div>
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



<div>
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

<div>
- And prepared for multiplex network construction by integrating node-to-node connection data.
</div>

```python
link = pd.concat([rice, wheat], ignore_index=True)
```

<div>
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

![Original Link](/assets/images/original%20link.png)

<br>

<div>
And I loaded the population and macro indicators dataset and modified them to have information that I need.
</div>

<div>
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

<div>
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

<div>
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

<div style="font-size: 36px;">
  <b>3. Analysis</b>
</div>

<div>
To analyze the datasets, I first looked at the desities of rice and wheat through 1986 to 2023.
</div>


![Density of Rice and Wheat](/assets/images/density%20of%20rice%20and%20wheat.png)

<div style="font-size: 18px;">
<b>• Graph explanation: </b>
</div>


<div>
- Y-axis (Density): Represents the ratio of actual trade connections to all possible connections in the network. A higher value means more countries are actively trading with each other.
</div>

<div>
- X-axis (Year): Spans from 1985 to 2022.
</div>

<br>

<div>
- A noticeable feature shows in between 1988 and 1922.
</div>

<div>
- This suggests a period of unusually high connectivity in rice trade, possibly due to policy shifts or global trade agreements.
</div>


<div>
- After those years, the rice trade network became significantly less dense and settled around a low, stable value, indicating fewer or more selective trade links.
</div>

<div>
- Since around 2010, the densities of rice and wheat networks have become nearly equal, indicating a possible convergence in the global trade structures for these two staple crops.
</div>

<div>
- The rice trade network was very connected at some times, but then broke into parts. The wheat network stayed stable and spread out. In recent years, both networks look similar, which shows they may be trading in the same way.
</div>

<br>

<div style="font-size: 18px;">
<b>• Out-degree and In-degree Distributions </b>
</div>

<div>
- I calculated weighted in-degrees and out-degrees for all countries to identify dominant importers/exporters for 2023 year. And checked the top five exporting and importing countries 
</div>

![indegree](/assets/images/indegree.png)

<div>
- So, In-degree means how many other countries a country imports from.
</div>

<div>
- France has the highest.
</div>

<div>
- Germany and Italy are also strong importers.
</div>

![outdegree](/assets/images/outdegree.png)

<div>
- Out-degree means how many other countries a country exports to.
</div>

<div>
- France exports to the most countries and plays a major role in global grain trade. And showing it is active in both export and import.
</div>

<div>
- Germany and Italy are also strong importers.
</div>

<div>
- Many European countries play a key role in connecting the global market.
</div>

<div>
- Some countries are active in both exporting and importing, showing they have multiple roles in international trade.
</div>
<br>

<div>
• And I looked at these 6 countries to see the patterns over 1986 to 2023. 
</div>

![indegreeover](/assets/images/indegreeover.png)

<div>
- France has always been the top importer with very high in-degree over the years.
</div>

<div>
- Germany and Italy also have steady in-degree values, showing that they consistently receive many connections in the trade network.
</div>

<div>
- Germany and Italy consistently receive many connections in the trade network.
</div>

<div>
- Overall, the top 3 countries remain stable, while the rest show smaller or more unstable import trends.
</div>

![outdegreeover](/assets/images/outdegreeover.png)

<div>
- France is again the highest.
</div>

<div>
- Germany, Italy, and Canada follow, with similar upward trends over time.
</div>

<div>
- Morocco also increases gradually, but its level remains relatively low.
</div>

<div>
- Overall, France stays the top country in both importing and exporting. Germany and Italy also play big roles. Kazakhstan and Morocco started small but are growing in the trade network.
</div>

<br>

<div style="font-size: 18px;">
<b>• Network Visualization: </b>
</div>

<div>
- Using force-directed, I visualized core trade networks, incorporating node size (betweenness centrality) and node color (modularity). This allowed identification of central actors and clustered communities.
</div>

![network](/assets/images/network.png)


<div>
- The size of each node shows its betweenness centrality – how important it is as a connector between other countries.
</div>

<div>
- Colors represent different modularity communities, meaning countries grouped by how closely they are connected within clusters.
</div>

<div>
- France, Germany, and Italy are central and large, meaning they play major roles in connecting global trade flows.
</div>

<div>
- Big and central countries like France and Germany connect many others. Countries with the same color often trade more with each other. Some countries are not very connected and are at the edges.
</div>

<br>

<div style="font-size: 18px;">
- Rice trade map
</div>





<iframe src="/assets/rice_trade_map.html" width="100%" height="700px" frameborder="0"></iframe>

<div>
- This map shows the global rice trade network using nodes and curved lines
</div>

<div>
- Red circles represent countries.
</div>

<div>
- The larger the circle, the more rice that country exports.
</div>

<div>
- Red curved lines indicate the trade routes.
</div>

<div>
- Thicker lines show higher trade volume.
</div>

<div>
- India, Thailand, and Vietnam are major exporters, as shown by their large red circles. They dominate rice exports globally.
</div>

<div>
- Africa and parts of the Middle East receive many trade links, suggesting they are key importers of rice.
</div>


<br>

<div style="font-size: 18px;">
- Wheat trade map
</div>

<iframe src="/assets/wheat_trade_map.html" width="100%" height="700px" frameborder="0"></iframe>

<div>
- This map shows the international wheat trade network using nodes and curved edges
</div>

<div>
- Blue circles represent countries.
</div>

<div>
- The bigger the circle, the more wheat that country exports
</div>

<div>
- Red curved lines show wheat trade flows between countries.
</div>

<div>
- The thicker the line, the larger the trade volume.
</div>

<div>
- Overall, This map is showing France, Germany, and Canada as major exporters. Red trade flows show how wheat moves mainly from Europe and North America to Africa and Asia, revealing the structure and direction of global wheat distribution.
</div>

