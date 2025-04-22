---
layout: post
title:  "Global Trade Networks through Rice and Wheat: A Data-Driven View of Global Flows"
---

<div style="text-align: center;">
    This project is from my class. Feel free to look around.
</div>

<div style="font-size: 24px;">
    Source
</div>

<p>
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2021) Deng et al.pdf" target="_blank">(2021) Deng et al.pdf.</a>
</p>

<div style="font-size: 24px;">
  <b>1. Summary of the reading</b>
</div>

<div>
  <b> • Research Question & Research Gap</b>
</div>

<div>
Research question: How does virtual water flow between countries, and how can we use social network analysis (SNA) to understand the global structure of this flow?
</div>

<br>

<div>
Research Gap: Most past research focused only on agricultural products and didn’t explore the whole-industry level or apply SNA to network characteristics.
</div>

<br>

<div>
  <b> • Method</b>
</div>

<div>
What is used: MRIO model (EORA26) + SNA (Density, Asymmetry, In/Out-Degree)
</div>

<br>

<div>
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

<div style="font-size: 24px;">
  <b>2. Data introduction</b>
</div>

<br>

<div>
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

<div>
  <b> • Major characteristics of the data</b>
</div>


<div>

</div>


<div>
</div>

<div>
</div>

<div>
  <b> • Data pre-processing</b>
</div>

<div>
</div>


<div>
</div>



<div>
</div>


<div>
</div>


<div>
</div>





```python
import networkx as nx
import pandas as pd

df = pd.read_csv("data.csv")
G = nx.from_pandas_edgelist(df, source="source", target="target")
```
