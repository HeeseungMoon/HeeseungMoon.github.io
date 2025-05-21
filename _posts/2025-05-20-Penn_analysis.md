---
layout: post
title:  "Analyzing Urban Communities in Pennsylvania"
---

<div style="font-size: 36px; text-align: center;">
    Source
</div>

<p style="text-align: center;">
  📄 <a href="https://github.com/HeeseungMoon/HeeseungMoon.github.io/raw/master/assets/(2023) Andris et al.pdf" target="_blank">(2023) Andris et al.pdf.</a>
</p>

<div style="font-size: 36px; text-align: center;">
  <b>1. Summary of the reading</b>
</div>

<div style="font-size: 18px; text-align: center;">
  <b> • Research Question & Research Gap</b>
</div>

<div style="text-align: center;">
- Research question: Are functional regions derived from human mobility and social networks more effective than state boundaries in containing the spread of COVID-19?
</div>

<br>

<div style="text-align: center;">
- Research Gap: While prior research has shown that reducing travel can slow the spread of infectious diseases, there has been little empirical evaluation of which types of geographic boundaries are most effective in practice.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Method</b>
</div>

<div style="text-align: center;">
- Commutes (LODES, 2015)
</div>

<div style="text-align: center;">
- GPS based trips (SafeGraph, Jan–Feb 2020)
</div>

<div style="text-align: center;">
- Migration flows (ACS, 2013–2017)
</div>

<div style="text-align: center;">
- Twitter comentions (2014–2015)
</div>

<div style="text-align: center;">
- Facebook friendships (Social Connectedness Index)
</div>

<div style="text-align: center;">
- COVID-19 case data (New York Times, 2020–2022)
</div>

<div style="text-align: center;">
- County adjacency network (for detecting within vs. between region pairs)
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Results + Interpretation</b>
</div>

<div style="text-align: center;">
- Commute based regions had the lowest COVID-19 transmission across boundaries and the most homogeneous case rates within.
</div>

<div style="text-align: center;">
- GPS-trip-based regions also performed well.
</div>

<div style="text-align: center;">
- Facebook-based regions were the least effective at defining boundaries for disease control.
</div>

<div style="text-align: center;">
- Statistical tests confirmed that commute regions had significantly higher internal case clustering and lower cross region transmission (p < 0.001).
</div>

<div style="text-align: center;">
- Interpretation: Functional regions based on how people actually move serve as natural barriers to disease spread and may be better suited than state lines for pandemic policy planning.
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
- These are the data that I used to analyze.
</div>

<div style="text-align: center;">
<a href="https://www.census.gov/topics/population/migration/guidance/migration-flows.html" target="_blank"> County-to-County Migration Flows data (Download)</a>
</div>


<div style="text-align: center;">
  <a href="/assets/data/R13859119_SL050.csv" target="_blank">R13859119_SL050.csv (Download)</a>
</div>

<div style="text-align: center;">
  <a href="/assets/data/R13859119.txt" target="_blank">R13859119.txt (Download)</a>
</div>

<div style="text-align: center;">
  <a href="/assets/data/CBSA__MSA__2019_US_SL310_Coast_Clipped.shp" target="_blank">
    CBSA (MSA) 2019 US ZIP (Download)
  </a>
</div>

<div style="text-align: center;">
  <a href="/assets/data/COUNTY_2019_US_SL050_Coast_Clipped.shp" target="_blank">
    COUNTY 2019 US ZIP (Download)
  </a>
</div>

<br>

<div style="text-align: center;">
- In this project, I used County-to-County Migration Flows data provided by the U.S. Census Bureau.
The data shows how many people moved between each county in the United States from 2005 to 2020.
</div>

<div style="text-align: center;">
- The shape files used in this project are all public data provided by U.S. government agencies, and are files that allow local boundaries to be represented on a map.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
  <b> • Data pre-processing</b>
</div>

```python
df = pd.read_excel('county-to-county-2016-2020-ins-outs-nets-gross.xlsx', sheet_name='Pennsylvania', header=None, skiprows=3, nrows=21668 )
```
<div style="text-align: center;">
- In this analysis, I wanted to know about Pennsylvania so, I used a sheet named 'Pennsylvania' to extract only migration flows from Pennsylvania.
</div>

<div style="text-align: center;">
- Skiprows=3 option skip the top description row,
</div>

<div style="text-align: center;">
- The nrows= 21668 limits data size for faster processing.
</div>


```python
df2 = df.iloc[:, :9]
```

<div style="text-align: center;">
- Slices only the front nine columns from the original data df and stores them in the new data frame df2.
</div>


```python
df2.columns= ['s_code_a', 'c_code_a', 's_code_b',
             'c_code_b', 's_name_a', 'c_name_a', 's_name_b',
            'c_name_b', 'b_to_a']
```

<div style="text-align: center;">
- I also changed the column names to make it easier to understand.
</div>

<div style="text-align: center;">
#'s_code_a' -> 'State Code of A'
</div>

<div style="text-align: center;">
#'c_code_a' -> 'County Code of A
</div>

<div style="text-align: center;">
#'r_code_b' -> Region Code of B
</div>

<div style="text-align: center;">
#'c_code_b' -> 'County Code of B
</div>

<div style="text-align: center;">
#'s_name_a' -> State Name of A
</div>

<div style="text-align: center;">
#'c_name_a' -> 'County Name of A
</div>

<div style="text-align: center;">
#'r_name_b' -> Region of B
</div>

<div style="text-align: center;">
#'c_name_b' -> County Name of B'
</div>

<div style="text-align: center;">
#'b_to_a'   -> Flow from B to A
</div>

```python
df2 = df2.rename(columns={'b_to_a':'weight'})
```

<div style="text-align: center;">
- And I wanted to see the flow so I changed "b_to_a" to "weight"
</div>

```python
link = df2[
    (df2['s_name_a'] == "Pennsylvania") &
    (df2['s_name_b'] == "Pennsylvania") &
    (df2['weight'] > 50)
]
```

<div style="text-align: center;">
- Both origin and destination are only left in Pennsylvania.
</div>

<div style="text-align: center;">
- Noise is removed leaving only meaningful flows except when the number of migrant populations is 50 or less.
</div>

![Link_shape](/assets/images/link_shape.png)

<div>
- Consequently, link.shape indicates that (931, 9) → 931 significant connections exist.
</div>

```python
g = nx.from_pandas_edgelist(
    link,
    source='c_code_a',
    target='c_code_b',
    edge_attr='weight'
)
```

<div style="text-align: center;">
- c_code_a and c_code_b are the codes of origin/destination counties, respectively
</div>

<div style="text-align: center;">
- weight is the number of people moving, used as a weight for the edge
</div>

<div style="text-align: center;">
- Use this code to create an undirected network graph g
</div>


```python
degree_dict = dict(g.degree(weight="weight"))
nx.set_node_attributes(g, degree_dict, 'degree')
```

<div style="text-align: center;">
- By calculating the degree of each node (county), you can find counties with active movement.
</div>

<div style="text-align: center;">
- Calculate the degree based on the weight and store it as a node property.
</div>


```python
node_labels = df2.groupby('c_name_a')['c_code_a'].first().to_dict()
nx.set_node_attributes(g, node_labels, name='c_code_a')
```

<div style="text-align: center;">
- Mapping numeric FIPS codes to human-readable county names.
</div>

<div style="text-align: center;">
- Create node_labels dictionary, give each node its name.
</div>

```python
labels = nx.get_node_attributes(g, 'c_code_a')
```

<div style="text-align: center;">
- Save each node's name as a labels variable for visualization
</div>


![Network2](/assets/images/network2.png)

<div style="text-align: center;">
- And draw the network.
</div>


```python
modularity_df = pd.DataFrame(
    [
        [k + 1, nx.community.modularity(g, communities[k])]
        for k in range(len(communities))
    ],
    columns=["k", "modularity"],
)
```

<div style="text-align: center;">
- communities[k]: k communities created by the Girvan–Newman algorithm
</div>

<div style="text-align: center;">
- nx.community.modularity(): Calculate the modularity score for the result of that community split
</div>

<div style="text-align: center;">
- The results are stored in modularity_df, and each row contains the number of communities (k) and the corresponding modularity value.
</div>


![modularity](/assets/images/modularity.png)

```python
modularity_df.plot.bar(
    x="k",
    color="#F2D140",
    title="Modularity Trend for Girvan-Newman Community Detection"
)
plt.tight_layout()
plt.show()
```

<div style="text-align: center;">
- Bar graph with the number of communities (k) set on the X-axis and the modularity value set on the Y-axis.
</div>

<div style="text-align: center;">
- The results show a U-shaped pattern in which the modularity value rises at a certain k and then decreases again.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
<b>• Interpretation: </b>
</div>

<div style="text-align: center;">
- Modularity peaks between approximately k = 30 and 40, indicating that the community structure is most pronounced in this section.
</div>

<div style="text-align: center;">
- Then, when k becomes too large, the network becomes over-divided, and the modularity becomes sharply lower.
</div>

<div style="text-align: center;">
- This demonstrates that the Girvan–Newman algorithm over-segmented the community reduces its structural significance.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
<b>• Degree Centrality Visualization </b>
</div>

<div style="text-align: center;">
- This visualization is based on degrees centrality in the inter county population movement network within Pennsylvania.
</div>

<div style="text-align: center;">
- Visually represents the relative importance of each county (node).
</div>



```python
degree = nx.degree_centrality(g)
```

<div style="text-align: center;">
- Degree centrality—Indicates how many other nodes one node is directly connected to.
</div>

<div style="text-align: center;">
- The more traveled counties, the more central they are, the more likely they are to serve as hubs within the network.
</div>



```python
nx.draw_networkx_nodes(
    g, pos,
    node_size=[v * 5000 for v in degree.values()],
    node_color=list(degree.values()),
    cmap=plt.cm.viridis
)
```

<div style="text-align: center;">
- Node size: bigger degree value is bigger
</div>

<div style="text-align: center;">
- Node color: Bright yellow for high centrality nodes and dark purple for low-centrality nodes
</div>


<div style="text-align: center;">
- cmap=viridis: yellow to purple color map
</div>

```python
nx.draw_networkx_edges(g, pos, alpha=0.3)
nx.draw_networkx_labels(g, pos, labels=nx.set_node_attributes(g, degree_dict, 'degree'), font_size=10)
```

<div style="text-align: center;">
- Visualize the connection structure by drawing a translucent edge.
</div>

<div style="text-align: center;">
- Label each node with a centrality value.
</div>

<div style="font-size: 18px; text-align: center;">
<b>• Interpretation: </b>
</div>

<div style="text-align: center;">
- The nodes located in the center of the graph are connected to many counties, so the degree centrality is very high → These are the main counties that act as hubs for movement.
</div>

<div style="text-align: center;">
- On the other hand, the nodes on the outskirts have less connection and smaller centrality and visual size.
</div>

<div style="text-align: center;">
- Bright colors and large nodes allow intuitive identification of critical network centers.
</div>

![Centrality](/assets/images/Centrality.png)

```python
node = pd.read_csv('R13859119_SL050.csv')
```

<div style="text-align: center;">
- R13859119_SL050.csv is a county-level socioeconomic statistics data provided by Social Explorer.
</div>

```python
node_attr = node.set_index("Geo_COUNTY").to_dict("index")
nx.set_node_attributes(g, node_attr)
```


<div style="text-align: center;">
- Set Geo_COUNTY for each county as the key, and assign the entire row to the graph node in dictionary form.
</div>

<div style="text-align: center;">
- These properties are directly linked to nodes in the g graph through nx.set_node_attributes().
</div>

<div style="font-size: 36px; text-align: center;">
  <b>3. Analysis</b>
</div>


```python
from community import community_louvain       
louv = community_louvain.best_partition(g, weight='weight')
```

<div style="text-align: center;">
- The community_louvain.best_partition() function takes network g as input and returns a dictionary that determines which community to assign each node to.
</div>

<div style="text-align: center;">
- weight='weight' means that we detect community structures based on the number of migrant populations.
</div>

```python
node['louv'] = node['Geo_COUNTY'].map(louv)
```

<div style="text-align: center;">
- The Louvain result (louv) is added to the node data frame based on the county code (Geo_COUNTY).
</div>

<div style="text-align: center;">
- The new luv column represents the community number to which each county belongs.
</div>

```python
import igraph as ig
import leidenalg as la

gg = ig.Graph.from_networkx(g)
coms = la.find_partition(gg, la.ModularityVertexPartition, weights='weight')
```

<div style="text-align: center;">
- graphics are libraries that can perform community detection faster than networkx.
</div>

<div style="text-align: center;">
- gg is a conversion of an existing NetworkX graph to an iGraph object.
</div>

<div style="text-align: center;">
- The find_partition() function runs the Leiden algorithm and returns a community list.
</div>

```python
leid = {}
for j in range(0, len(coms)):
    leid.update({member: j for member in coms[j]})
```

<div style="text-align: center;">
- The coms list contains nodes that belong to each community.
</div>

<div style="text-align: center;">
- Based on this, create a pre-read that stores the node (FIPS code) → community ID.
</div>

```python
node['leid'] = [leid.get(node) for node in node['Geo_COUNTY']]
```

<div style="text-align: center;">
- Add the community number (leid) to which each county (Geo_COUNTY) belongs as a new column to the data frame.
</div>

```python
map = gpd.read_file("COUNTY_2019_US_SL050_Coast_Clipped.shp")
```

<div style="text-align: center;">
- COUNTY_2019_US_SL050_Coast_Clipped.shp is a shapefile containing county boundaries across the United States.
</div>


```python
map['STATEFP'] = map['STATEFP'].astype(int)
map = map[(map['STATEFP'] == 42)]
```

<div style="text-align: center;">
- STATEFP is the state's unique code. Pennsylvania code is 42.
</div>

<div style="text-align: center;">
- This limits the area to be analyzed to Pennsylvania.
</div>

```python
merged = pd.merge(map, node, left_on='COUNTYFP', right_on='Geo_COUNTY')
```

<div style="text-align: center;">
- Merge based on COUNTYFP in map data (map) and Geo_COUNTY in statistical data (node)
</div>

<div style="text-align: center;">
- Merged, which is the result of the merge, contains the geometry + attribute information by county.
</div>

<div style="text-align: center;">
- This data is used for all subsequent analyses, including visualization, statistical analysis, and community classification.
</div>

<br>

<div style="font-size: 18px; text-align: center;">
<b>• Community Visualization (Louv) </b>
</div>

![louv](/assets/images/louv.png)

<div style="text-align: center;">
- This map is a color-coded visualization of which Louvain community each county in Pennsylvania belongs to.
</div>

<div style="text-align: center;">
- Based on the migration flow of people, it is possible to see the structures in which regions showing geographically similar patterns of movement are grouped together.
</div>

<div style="text-align: center;">
- There are 4 communities. Adjacent counties clearly tend to be grouped together in the same color, reflecting the practical connectivity of the migration network base between regions.
</div>

<br>

![leid](/assets/images/leid.png)

<div style="text-align: center;">
- This map is the result of visualizing the detected community configuration over Pennsylvania geography using the Leiden algorithm.
</div>

<div style="text-align: center;">
- Compared to previous Louvain, we show a more sophisticated and connection-oriented community structure.
</div>


<div style="text-align: center;">
- There are areas where communities are more geographically dispersed, or where they appear to be independently formed reflecting close migration patterns.
</div>

<div style="text-align: center;">
- Some regions are more delicately isolated than Louvain, as Leiden aims for higher modularity optimization and maintaining internal connectivity.
</div>


![metro_louv](/assets/images/metro_penn_louv.png)

<div style="text-align: center;">
- The red border is an officially designated Metropolitan Statistical Area (MSAs)**, an administrative unit based on the labor market and economic sphere.
</div>


<div style="text-align: center;">
- By comparing community and MSA boundaries, we can evaluate how much the administrative districts match the movement patterns of real people.
</div>

<br>

![metro_leid](/assets/images/metro_penn_leid.png)

<div style="text-align: center;">
- The Leiden community is an automatically detected structure based on movement patterns and connectivity, and is formed differently from **administrative compartment (MSA)**.
</div>


<div style="text-align: center;">
- For example, the neighborhoods of Reading and Philadelphia are grouped into one community (light green), but the MSA is separate.
</div>


<div style="text-align: center;">
- The northern Bloomsburg-Berwick region is detected as an independent community (yellow) in the Leiden algorithm, revealing the isolation of the migration.
</div>

<br>

![income_louv](/assets/images/income_louv.png)

<div style="text-align: center;">
- Community 2 has the highest median income, and its distribution is broad.
</div>

<div style="text-align: center;">
- Some counties have very high incomes above $120,000
</div>

<div style="text-align: center;">
- Community 1 and Community 3 have relatively low income levels, with median values between $60 and $70,000
</div>

<div style="text-align: center;">
Community 0 shows a moderate income distribution, with no significant outliers seen
</div>

<br>

![doc_louv](/assets/images/doc_louv.png)

<div style="text-align: center;">
- It shows that Community 2 has the highest percentage of Ph.D.s and is likely to be a community with a higher overall academic level.
</div>

<div style="text-align: center;">
- Community 0, 1, and 3 have similar low PhD rates, but some outliers indicate high PhD rates in certain counties.
</div>

<div style="text-align: center;">
- In Community 1, a ratio close to zero is also observed, resulting in a large overall variance.
</div>

<br>

![master_louv](/assets/images/master_louv.png)

<div style="text-align: center;">
- Community 2 has an overwhelming percentage of master's degrees.
</div>

<div style="text-align: center;">
- Both the median and overall distributions represent high levels.
</div>

<div style="text-align: center;">
- Communities 0 and 3 remain moderate.
</div>

<div style="text-align: center;">
- Community 1 has the lowest median, and some counties have very low median values
</div>

<br>

![prof_louv](/assets/images/prof_louv.png)

<div style="text-align: center;">
- Community 2 has a significantly higher percentage of professional degree holders than other communities.
</div>

<div style="text-align: center;">
- Where both the box median and overall distribution are highest
</div>

<div style="text-align: center;">
- It can be interpreted as a region with a high proportion of highly educated people and professional employment
</div>

<div style="text-align: center;">
- Community 0, 1, and 3 are similar, but the median value of Community 1 is somewhat lower.
</div>

<div style="text-align: center;">
- Community 3 also has some outliers, so there is a possibility of heterogeneity within the region
</div>

<br>

<div style="text-align: center;">
- By looking at these data, I could find some cahracteristics.
</div>


<div style="text-align: center;">
- Most wealthier communities tend to have higher levels of education and employment rates.
</div>


<div style="text-align: center;">
But they also tend to have more unemployment. Probably, their expectations for job is higher because they did get higher education degree.
</div>

<div style="font-size: 36px; text-align: center;">
  <b>Conclusion</b>
</div>

<div style="text-align: center;">
In conclusion, there were distinct differences in socioeconomic characteristics among communities, including education level, income, and employment rate, suggesting that it may be more realistic to consider network-based communities than administrative districts in establishing space-based policies.
</div>

<div style="text-align: center;">
Interestingly, the higher the education level and employment rate, the higher the unemployment rate. This suggests that the highly educated population may be more sensitive to job search conditions due to high expectations, or may be temporarily inactive considering the quality of their jobs.
</div>


<div style="text-align: center;">
Therefore, rather than simply judging the economic vitality of a region based on unemployment figures, it is necessary to interpret it in the context of the level of education or human capital in the region.
</div>