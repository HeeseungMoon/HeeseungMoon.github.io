{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Project #0 Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Project Outline**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 1. Define Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2. Data Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3. Explanatory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4. Data Pre-processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 5. Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 6. Model Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 6.5 Model Modification"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 7. Conclusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "# packages for data manipulation\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.patches as mpatches"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Problem Defination"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "predict what the species of a new data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Data Preparation: skip"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal_length  sepal_width  petal_length  petal_width species\n",
       "0           5.1          3.5           1.4          0.2  setosa\n",
       "1           4.9          3.0           1.4          0.2  setosa\n",
       "2           4.7          3.2           1.3          0.2  setosa\n",
       "3           4.6          3.1           1.5          0.2  setosa\n",
       "4           5.0          3.6           1.4          0.2  setosa"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris = sns.load_dataset('iris')\n",
    "iris.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "def hist_graphing(variable, bin):\n",
    "    plt.figure(figsize=(6,6))\n",
    "    sns.histplot(data = iris, x = variable, bins = bin, hue = 'species', alpha = 0.3)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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6quO+vHjx6tixYqaOHGiDhw4oHLlyqlp06Z66aWXJEkVKlTQjz/+qOHDh6tNmzby8PBQ48aNFRkZmW0/N5uep6enRo0apYMHD8rHx0etW7fWwoUL8309FTSbVUhuCWaz2fTNN99ke9vTS5cu6Z///KeOHDmi1atX3zBUZHekIigoSGfPnr1hHfJuy5YtatasmV6cNF3BNcNcqk04sFdvjByg2NhYNW3aNJ86BJAXFy9eVHx8vEJDQ/P8NMv89MQTT+jMmTNZLv5Hzm60Xe12u/z9/Z3+Di30RyouXbqkhx56SIcOHXL8rfCNeHl5OS6MgXsE1wxTWP2G7m4DAFDACnWouBoo9u7dq1WrVqlChQrubgkAAOTAraEiJSVF+/btc7yPj4/Xtm3bVL58eQUGBurBBx/Uli1btGTJEmVkZOj48eOSpPLly8vT09NdbQMACrG5c+e6u4VblltDxebNm9WuXTvH+6FDh0qSoqOj9eqrr2rx4sWSpMaNG2epW7Vqldq2bVtQbQIAACe4NVS0bdv2ho+OLSTXkAIAACfwlFIAAGAEoQIAABhBqAAAAEYQKgAAgBGF+j4VAICiIS/P/ckNnhVUOBEqAAB5ktfn/uRGQT0r6ODBgwoNDdXWrVuvu70BrkeoAADkSVJSklJTU3P13J/cuPqsoKSkJI5WFDKECgCAEYX5uT9fffWVxo4dq3379snX11dNmjTRt99+q9KlS+vjjz/WlClTFB8frxo1amjQoEHq37+/JCk0NFSS1KRJE0lSmzZttHr1amVmZuq1117TrFmzHI9cf+ONN9S5c2dJUnp6uoYOHaqvv/5aycnJqly5sp555hmNGjVKkvT2229rzpw5OnDggMqXL6/u3bvrzTffVJkyZdywdswhVAAAirXExEQ98sgjevPNN/WPf/xD586d088//yzLsjR//nzFxMTo/fffV5MmTbR161b169dPpUuXVnR0tH799Ve1bNlSK1asUIMGDRyPiJg2bZqmTJmiDz/8UE2aNNHs2bN133336bffflNYWJjeffddLV68WF988YWCg4N1+PBhHT582NFTiRIl9O677yo0NFQHDhxQ//79NWLECH3wwQfuWk1GECoAAMVaYmKiLl++rB49eigkJESSFBERIUkaM2aMpkyZoh49eki6cmRi9+7d+vDDDxUdHa2KFStKkipUqKAq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1yuvvKJ//OMf2rp1qxo3bnzd6Y+5c+dqyJAh+vzzzzVkyBAdPnxYd911l+bMmaPAwMAs87p6+iMzM1OTJ0/WrFmzdPjwYVWuXFlPP/20Xn75ZUnSyJEj9c033+jIkSOqUqWKoqKiFBMTo1KlShXoOs0JF2oCAIq1nj176s8//9SqVascw06fPq2lS5cqKioq25qVK1dqz549Wr58uZYsWSK73a7u3bsrIiJCW7Zs0fjx4zVy5Mibzjs1NVWTJ0/Wv//9b61Zs0YJCQkaNmxYjuOPGjVKb7zxhkaPHq3du3drwYIFqly5suPzsmXLau7cudq9e7emTZumjz76SO+8844LayN/caQCAFCs3XbbberSpYsWLFig9u3bS5K++uorBQQEqF27dvr555+vqyldurQ+/vhjx2mPmTNnymaz6aOPPpK3t7fq16+vo0ePql+/fjec96VLlzRz5kzVqlVLkjRw4ECNGzcu23HPnTunadOm6f3331d0dLQkqVatWrrrrrsc47zyyiuO/69Ro4aGDRumhQsXasSIES6skfzDkQoAQLEXFRWlr7/+WmlpaZKk+fPnq1evXllOh/xVREREluso9uzZo4YNG8rb29sxrGXLljedr6+vryNQSFJgYKBOnjyZ7bhxcXFKS0tzBJ/sfP7554qMjFSVKlVUpkwZvfLKK0pISLhpHwWFUAEAKPa6d+8uy7L03Xff6fDhw/r5559zPPUhXTlSYcK11zrYbLYc/9rDx8fnhtNav369oqKidO+992rJkiXaunWrXn75ZaWnpxvp1QRCBQCg2PP29laPHj00f/58ffbZZ6pbt66aNm3qdH3dunW1c+dOx5EOSdq0aZPRHsPCwuTj46OVK1dm+/m6desUEhKil19+Wc2bN1dYWJgOHTpktIe8IlQAAG4JUVFR+u677zR79uwbHqXIzqOPPqrMzEz961//UlxcnJYtW6bJkydLunL0wQRvb2+NHDlSI0aM0Keffqr9+/drw4YN+r//+z9JV0JHQkKCFi5cqP379+vdd9/VN998Y2TepnChJgDAiEP7CuZfzbmdz9///neVL19ee/bs0aOPPupSrZ+fn/773//q2WefVePGjRUREaGYmBg9+uijWa6zyKvRo0erZMmSiomJ0bFjxxQYGKhnnnlGknTffffp+eef18CBA5WWlqauXbtq9OjRevXVV43NP69sVnG6lVc27Ha7/P39dfbsWfn5+bm7nWJty5YtatasmT748n8Kq9/Qpdq9u3eof8+Oio2NdemQJICCc/HiRcXHxys0NDTLF2lRv6Nmbs2fP199+vTR2bNnb3o9RGGW03aVXP8O5UgFACBPgoODFRcXV2Sf/eGsTz/9VDVr1lS1atW0fft2jRw5Ug899FCRDhSmESoAAHkWHBzs1qMGBeH48eOKiYnR8ePHFRgYqJ49e+r11193d1uFCqECAAAnjBgxotDcZKqw4q8/AACAEW4NFWvWrFH37t1VtWpV2Wy2654nb1mWYmJiFBgYKB8fH3Xo0EF79+51T7MAAOCG3Boqzp8/r0aNGmn69OnZfv7mm2/q3Xff1cyZM7Vx40aVLl1anTp10sWLFwu4UwDAVZmZme5uAQaZ3J5uvaaiS5cu6tKlS7afWZalqVOn6pVXXtH9998v6cqVt5UrV9aiRYvUq1evgmwVAG55np6eKlGihI4dO6aKFSvK09PT2I2fUPAsy1J6erpOnTqlEiVKZHnWSW4V2gs14+Pjdfz4cXXo0MExzN/fX3fccYfWr1+fY6hIS0vLchtVu92e770CzkhISMjVn9y540/ngOyUKFFCoaGhSkxM1LFjx9zdDgzx9fVVcHBwjg9Xc0WhDRXHjx+XpCzPkb/6/upn2Zk4caLGjh2br70BrsrLzYEKw01+gKs8PT0VHBysy5cvKyMjw93tII88PDxUsmRJY0ecCm2oyK1Ro0Zp6NChjvd2u11BQUFu7AiQkpKSlJqaqnlvDVd4Ted/HuMOHFbv4W8pKSmJUIFCw2azqVSpUtc9gRMotKGiSpUqkqQTJ04oMDDQMfzEiRNq3LhxjnVeXl7y8vLK7/aAXAmvGaSmDWq7uw0AyBeF9j4VoaGhqlKlSpZHwNrtdm3cuFGtWrVyY2cAACA7bj1SkZKSon379jnex8fHa9u2bSpfvryCg4M1ZMgQvfbaawoLC1NoaKhGjx6tqlWr6oEHHnBf0wAAIFtuDRWbN29Wu3btHO+vXgsRHR2tuXPnasSIETp//rz+9a9/6cyZM7rrrru0dOlSo4+ZBQAAZrg1VLRt21Y3evK6zWbTuHHjNG7cuALsCgAA5EahvaYCAAAULYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgRKEOFRkZGRo9erRCQ0Pl4+OjWrVqafz48bIsy92tAQCAa5R0dwM3MmnSJM2YMUOffPKJGjRooM2bN6tPnz7y9/fXoEGD3N0eAAD4i0IdKtatW6f7779fXbt2lSTVqFFDn332mX799Vc3dwYAAK5VqEPFnXfeqVmzZumPP/5QnTp1tH37dv3yyy96++23c6xJS0tTWlqa473dbi+IVmFIXFycyzUBAQEKDg7Oh24AAK4o1KHixRdflN1uV7169eTh4aGMjAy9/vrrioqKyrFm4sSJGjt2bAF2CRNOnzopSerdu7fLtb6+voqLiyNYAICbFepQ8cUXX2j+/PlasGCBGjRooG3btmnIkCGqWrWqoqOjs60ZNWqUhg4d6nhvt9sVFBRUUC0jl1LOnZUk9XkhRi3+dpfTdQkH9uqNkQOUlJREqAAANyvUoWL48OF68cUX1atXL0lSRESEDh06pIkTJ+YYKry8vOTl5VWQbcKgwOo1FFa/obvbAADkQqH+k9LU1FSVKJG1RQ8PD2VmZrqpIwAAkJNCfaSie/fuev311xUcHKwGDRpo69atevvtt9W3b193twYAAK5RqEPFe++9p9GjR6t///46efKkqlatqqeffloxMTHubg0AAFyjUIeKsmXLaurUqZo6daq7WwEAADdRqK+pAAAARQehAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgBKECAAAYQagAAABGECoAAIARhAoAAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYAShAgAAGEGoAAAARhAqAACAEYQKAABgREl3NwAAyF5CQoKSkpJcrgsICFBwcHA+dATcGKECAAqhhIQEhYeHKzU11eVaX19fxcXFESxQ4AgVAFAIJSUlKTU1VS9Ne0khtUOcrju075AmDJ6gpKQkQgUKHKECAAqxkNohqhNRx91tAE7hQk0AAGAEoQIAABhBqAAAAEYQKgAAgBGECgAAYESuQkXNmjX1559/Xjf8zJkzqlmzZp6bAgAARU+uQsXBgweVkZFx3fC0tDQdPXo0z00BAICix6X7VCxevNjx/8uWLZO/v7/jfUZGhlauXKkaNWoYaw4AABQdLoWKBx54QJJks9kUHR2d5bNSpUqpRo0amjJlirHmAABA0eFSqMjMzJQkhYaGatOmTQoICMiXpgAAQNGTq9t0x8fHm+4DAAAUcbl+9sfKlSu1cuVKnTx50nEE46rZs2fnuTEAAFC05CpUjB07VuPGjVPz5s0VGBgom81mui8AAFDE5CpUzJw5U3PnztVjjz1muh8AAFBE5eo+Fenp6brzzjtN9wIAAIqwXIWKp556SgsWLDDdCwAAKMJydfrj4sWLmjVrllasWKGGDRuqVKlSWT5/++23jTQHAACKjlyFih07dqhx48aSpF27dmX5jIs2AQC4NeUqVKxatcp0HwAAoIjj0ecAAMCIXB2paNeu3Q1Pc/z444+5bggAABRNuQoVV6+nuOrSpUvatm2bdu3add2DxgAAwK0hV6HinXfeyXb4q6++qpSUlDw1BAAAiiaj11T07t2b534AAHCLMhoq1q9fL29vb5OTBAAARUSuTn/06NEjy3vLspSYmKjNmzdr9OjRRhoDAABFS65Chb+/f5b3JUqUUN26dTVu3Dh17NjRSGMAAKBoyVWomDNnjuk+AABAEZerUHFVbGys4uLiJEkNGjRQkyZNjDQFAACKnlyFipMnT6pXr15avXq1ypUrJ0k6c+aM2rVrp4ULF6pixYomewQAAEVArv7647nnntO5c+f022+/6fTp0zp9+rR27dolu92uQYMGGW3w6NGj6t27typUqCAfHx9FRERo8+bNRucBAADyLldHKpYuXaoVK1YoPDzcMax+/fqaPn260Qs1k5OTFRkZqXbt2umHH35QxYoVtXfvXt12223G5gEAAMzIVajIzMxUqVKlrhteqlQpZWZm5rmpqyZNmqSgoKAsF4aGhoYamz4AADAnV6Hi73//uwYPHqzPPvtMVatWlXTlNMXzzz+v9u3bG2tu8eLF6tSpk3r27KmffvpJ1apVU//+/dWvX78ca9LS0pSWluZ4b7fbjfUDwLyEhAQlJSW5XJeWliYvL69czTMgIEDBwcG5qgWQs1yFivfff1/33XefatSooaCgIEnS4cOHdfvtt2vevHnGmjtw4IBmzJihoUOH6qWXXtKmTZs0aNAgeXp65vjgsokTJ2rs2LHGegCQfxISEhQeHq7U1FSXa22SrFzO19fXV3FxcQQLwLBchYqgoCBt2bJFK1as0O+//y5JCg8PV4cOHYw2l5mZqebNm2vChAmSpCZNmmjXrl2aOXNmjqFi1KhRGjp0qOO93W53BB8AhUtSUpJSU1M1763hCq/p/H66ev1mvTD5U/Ud1ld3tLvDpXke2ndIEwZPUFJSEqECMMylUPHjjz9q4MCB2rBhg/z8/HTPPffonnvukSSdPXtWDRo00MyZM9W6dWsjzQUGBqp+/fpZhoWHh+vrr7/OscbLyyvXh0QBuEd4zSA1bVDb6fEPHzkiSQoMClSdiDr51RYAF7n0J6VTp05Vv3795Ofnd91n/v7+evrpp/X2228bay4yMlJ79uzJMuyPP/5QSEiIsXkAAAAzXAoV27dvV+fOnXP8vGPHjoqNjc1zU1c9//zz2rBhgyZMmKB9+/ZpwYIFmjVrlgYMGGBsHgAAwAyXQsWJEyey/VPSq0qWLKlTp07luamrWrRooW+++UafffaZbr/9do0fP15Tp05VVFSUsXkAAAAzXLqmolq1atq1a5dq187+3OeOHTsUGBhopLGrunXrpm7duhmdJgAAMM+lIxX33nuvRo8erYsXL1732YULFzRmzBgCAAAAtyiXjlS88sor+s9//qM6depo4MCBqlu3riTp999/1/Tp05WRkaGXX345XxoFAACFm0uhonLlylq3bp2effZZjRo1SpZ15dYzNptNnTp10vTp01W5cuV8aRQAABRuLt/8KiQkRN9//72Sk5O1b98+WZalsLAwHvIFAMAtLld31JSk2267TS1atDDZCwAAKMJculATAAAgJ4QKAABgBKECAAAYQagAAABGECoAAIAR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7bff8rlDM/bu3auqVauqZs2aioqKUkJCQo7jrl+//rr10alTpxwfcV/YuLKsUtHdposXL1bz5s3Vs2dPVapUSU2aNNFHH310w5qiuG1zs5xXNW7cWIGBgbrnnnu0du3afO7UrPT0dM2bN099+/bN8V/kRXF7ZseZZZWK7r565513auXKlfrjjz8kSdu3b9cvv/yiLl265FhjYtsSKgrYwoULtWXLFk2cONGp8evWravZs2fr22+/1bx585SZmak777xTR44cyedO8+aOO+7Q3LlztXTpUs2YMUPx8fFq3bq1zp07l+34x48fv+4uqZUrVy7056Ml15e1qG5TSTpw4IBmzJihsLAwLVu2TM8++6wGDRqkTz75JMeanLat3W7XhQsX8rvlXMnNcgYGBmrmzJn6+uuv9fXXXysoKEht27bVli1bCrDzvFm0aJHOnDmjJ554IsdxivK++lfOLGtR3ldffPFF9erVS/Xq1VOpUqXUpEkTDRkyRFFRUTnWGNlXXTuggrxISEiwKlWqZG3fvt0x7GanP66Vnp5u1apVy3rllVfyocP8k5ycbPn5+Vkff/xxtp+XKlXKWrBgQZZh06dPtypVqlQQ7Rl1s2W9VlHapqVKlbJatWqVZdhzzz1n/e1vf8uxJiwszJowYUKWYd99950lyUpNTc2XPvMqN8uZnbvvvtvq3bu3ydbyVceOHa1u3brdcJzisq86s6zXKkr76meffWZVr17d+uyzz6wdO3ZYn376qVW+fHlr7ty5OdaY2Fc5UlGAYmNjdfLkSTVt2lQlS5ZUyZIl9dNPP+ndd99VyZIls1xAlJOriXPfvn0F0LE55cqVU506dXLsu0qVKi494r4wu9myXqsobdPAwEDVr18/y7Dw8PAbnu7Jadv6+fnJx8cnX/rMq9wsZ3ZatmxZJLarJB06dEgrVqzQU089dcPxisO+6uyyXqso7avDhw93HK2IiIjQY489pueff/6GR8lN7KuEigLUvn177dy5U9u2bXO8mjdvrqioKG3btk0eHh43nUZGRoZ27typwMDAAujYnJSUFO3fvz/Hvlu1apXlEfeStHz58iL5iPubLeu1itI2jYyMzHIlvCT98ccfCgkJybGmKG7b3CxndrZt21YktqskzZkzR5UqVVLXrl1vOF5R3J7XcnZZr1WU9tXU1FSVKJH1K97Dw0OZmZk5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",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "hist_graphing('sepal_length', 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Pj5ekydPVnJysl577TWNHTtWH374YYF1jBgxQi+88IJ27dql2NhY9ezZU7/++quky7dYr1u3rj7//HPt379f8fHxeumll/TZZ5+5XGv79u117tw57dq1S5K0fv16BQUFFQgE69evv+pGfld89tlnGj9+vF577TXt2LFDwcHBBcLTiy++qEceeUT33nuv0tPTlZ6ernbt2jlef/nll/Xiiy8qKSlJDRo0UN++fXXp0iWX3weA33HLPWoBlFqJiYl2SfYjR45c9dptt91mX7x4cYFlkyZNssfGxtrtdrs9NTXVLsk+depUx+sXL160161b1z5t2rRCxxw8eLD9j3/8o+N5//797Q888IBT9bZs2dL+xhtv2O12u71Xr172yZMn2728vOznzp2zHzt2zC7J/tNPP9ntdrt9wYIF9sDAQEff2NhY+6BBgwqs74477rA3a9bsurVceZ/vv/++Y9m+ffvskuzJyclO1Q2gcOzZAMq5Zs2aqXPnzoqJidHDDz+s9957T5mZmbpw4YIOHz6sp556SlWrVnU8Xn31VR0+fLjAOmJjYx3/rlSpklq3bq3k5GTHstmzZ6tVq1aqVauWqlatqvnz58tqtRap3g4dOighIUF2u10bN25U7969FR0dre+++07r169XSEiIoqKirtk3OTlZd9xxR6G130jTpk0d/w4ODpZ0+bbyAIqHE0SBcq5ixYpavXq1Nm/erP/85z9655139PLLL+vf//63JOm999676gO6YsWKTq9/yZIlevHFFzVjxgzFxsbK399fb7zxhrZt21akejt27KgPPvhAu3fvVuXKldWoUSN17NhRCQkJyszMVIcOHYq0XmdUrlzZ8W+LxSLp8mEiAMXDng3gJmCxWBQXF6cJEyZo165d8vLy0qZNmxQSEqKUlBTVr1+/wCMiIqJA/61btzr+fenSJSUmJio6OlqStGnTJrVr106DBg1SixYtVL9+/av2jLjiynkbb731liNYXAkbCQkJhZ6vIUnR0dFXhZz/rl2SvLy8HCegAigZ7NkAyrlt27ZpzZo16tq1q2655RZt27ZNv/zyi6KjozVhwgQ999xzCgwM1L333iubzaYdO3YoMzNTw4cPd6xj9uzZioqKUnR0tN566y1lZmbqySeflCRFRUXpo48+0qpVqxQREaF//vOf2r59+1WBxVnVq1dX06ZNtWjRIr377ruSpLvuukuPPPKILl68eN09G0OHDtUTTzyh1q1bKy4uTosWLdK+ffsUGRnpaFOvXj2tWrVKBw4cUM2aNRUYGFikOgE4j7ABlHMBAQHasGGDZs6cqaysLIWHh2vGjBnq3r27JMnPz09vvPGGRowYoSpVqigmJkbDhg0rsI6pU6dq6tSpSkpKUv369bV8+XIFBQVJkp5++mnt2rVLjz76qCwWi/r27atBgwbpm2++KXLNHTp0UFJSkmMvRo0aNdS4cWOdPHlSDRs2LLTfo48+qsOHD2vkyJHKycnRH//4Rz377LNatWqVo83AgQOVkJCg1q1b6/z581q3bp3q1atX5FoB3JjFbrfbPV0EgNLpyJEjioiI0K5du254yW8AKAznbAAAAKMIGwBKjNVqLfA1298/ivp1WQClG4dRAJSYS5cu6ciRI4W+Xq9ePVWqxKlkQHlD2AAAAEZxGAUAABhF2AAAAEYRNgAAgFGEDQAAYBRhAwAAGEXYAAAARhE2AACAUYQNAABg1P8HDss6oKbpTAkAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "hist_graphing('sepal_width', 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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eyYqkPCGUYqKbe7qGAAAFMIVMAEAgCHsmQAAVCqpqamujnDdYc8EAAAwhDIBAAAMoUwAAABDKBMAAMAQygQAADCEMgEAAAzho6EAAMMcvX+Qo7jvkHtxaZlISkrS559/rj179sjHx0c333yzpkyZUugmKl27dtWaNWsKjXv66ac1e/bs8o4LACiCkfsHOaq87jt06NAhRUZG6scff1TLli2duq6KzKVlYs2aNRo4cKDatWunixcv6qWXXlL37t21e/duVa9e3T7fk08+qQkTJtif+/r6uiIuAKAIVqtVubm5pb5/kKOu3HfIarWyd8JNuLRMLF26tNDz1NRU1a5dW2lpaercubN9uq+vr0JCQso7HgCgFNz5/kELFy7U+PHjtX//fvn6+qpVq1b68ssvVb16df3rX//StGnTlJ6ergYNGuj555/Xc889J0mKjIyUJLVq1UqS1KVLF61evVoFBQV69dVXNWfOHPutzydPnqyePXtKkvLz8zVs2DB99tlnOnPmjOrUqaNnnnlGCQkJkqQ33nhDKSkpOnjwoAIDA9WrVy+99tprqlGjhgu2jnFudQJmZmamJCkwMLDQ9Hnz5ikoKEg33XSTEhISynVXGgCgYsvIyNBDDz2kxx57TGazWatXr1bv3r1ls9k0b948JSYmauLEiTKbzZo0aZLGjBmj999/X5L0ww8/SJKWL1+ujIwMff7555KkGTNmaNq0aZo6dap27NihHj166K677tK+ffskSW+99ZYWL16sTz75RHv37tW8efPUoEEDe6YqVarorbfe0q5du/T+++9r5cqVGjlyZPlumDLkNidgFhQUaOjQoYqLi9NNN91kn/7www8rIiJCdevW1Y4dOzRq1Cjt3bvX/gP9o7y8POXl5dmfZ2VlOT07AMB9ZWRk6OLFi+rdu7ciIiIkSc2aNZMkjR07VtOmTVPv3r0lXd4TsXv3br3zzjvq37+/goODJUm1atUqtId86tSpGjVqlPr06SNJmjJlilatWqXp06crOTlZFotFUVFRuuWWW2QymezrvWLo0KH2rxs0aKBXX31VzzzzjN5++22nbQdncpsyMXDgQO3cuVP//e9/C01/6qmn7F83a9ZMoaGh6tatmw4cOKBGjRr9aTlJSUkaP3680/MCACqGFi1aqFu3bmrWrJl69Oih7t276/7775enp6cOHDigxx9/XE8++aR9/osXLyogIKDY5WVlZenYsWOKi4srND0uLk7bt2+XJA0YMEB//etfFR0drZ49e+rOO+9U9+7d7fMuX75cSUlJ2rNnj7KysnTx4kWdP39eubm5FfK8QLc4zDFo0CAtWbJEq1atUv369a86b4cOHSRJ+/fvL/L1hIQEZWZm2h9Hjhwp87wAgIrDw8NDy5Yt0zfffKPY2FjNnDlT0dHR2rlzpyTp3Xff1bZt2+yPnTt3auPGjYbW2bp1a6Wnp+uVV17Rb7/9pr///e+6//77JV3+hMidd96p5s2b67PPPlNaWpqSk5MlXT7XoiJy6Z4Jm82mwYMH64svvtDq1avtJ7pczbZt2yRJoaGhRb7u5eUlLy+vsowJAKjgTCaT4uLiFBcXp8TEREVERGj9+vWqW7euDh48qL59+xY5ztPTU5J06dIl+zR/f3/VrVtX69evV5cuXezT169fr/bt2xea78EHH9SDDz6o+++/Xz179tTp06eVlpamgoICTZs2TVWqXP4//SeffOKMb7vcuLRMDBw4UPPnz9eXX34pPz8/HT9+XJIUEBAgHx8fHThwQPPnz9cdd9yhWrVqaceOHXrhhRfUuXNnNW/unmcMAwDcy6ZNm7RixQp1795dtWvX1qZNm+yfwBg/fryef/55BQQEqGfPnsrLy9OWLVt05swZDRs2TLVr15aPj4+WLl2q+vXry9vbWwEBARoxYoTGjh2rRo0aqWXLlkpJSdG2bds0b948SZc/rREaGqpWrVqpSpUq+vTTTxUSEqKaNWuqcePGunDhgmbOnKlevXpp/fr1Ff7aSS4tE7NmzZJ0+cJUv5eSkqIBAwbI09NTy5cv1/Tp05WTk6OwsDDdd999Gj16tAvSAgCuxnJwn1uux9/fX2vXrtX06dOVlZWliIgITZs2TfHx8ZIuX37g9ddf14gRI1S9enU1a9bMfoJk1apV9dZbb2nChAlKTExUp06dtHr1aj3//PPKzMzUiy++qJMnTyo2NlaLFy9WVNTl62z4+fnptdde0759++Th4aF27drp66+/VpUqVdSiRQu98cYbmjJlihISEtS5c2clJSXpkUceKdPtVJ5MNpvN5uoQzpSVlaWAgABlZmbK39/f1XEcsnXrVrVp00Zvf/pdqT7DvW/3Dj33QHelpaWpdevWTkwI4Hpw/vx5paenKzIyUt7e3vbplfkKmNeD4n6uUsnfQ93m0xwAgIopPDxcZrOZe3NcxygTAADDwsPDeXO/jrnFR0MBAEDFRZkAAACGUCYAAIAhlAkAAGAIZQIAABhCmQAAAIZQJgAAgCFcZwIAYJjFYuGiVb9z6NAhRUZG6scff1TLli3dbnlljTIBADCEy2n/WVhYmDIyMhQUFOTqKOWCMgEAMMRqtSo3N1cfvj5CMQ3DnL4+88Ej6jfidVmtVpeViQsXLqhatWrFvu7h4aGQkJByTHRt+fn59luqlzXOmQAAlImYhmFq3bSx0x+lLSxz5sxR3bp1VVBQUGj63Xffrccee0yS9OWXX6p169by9vZWw4YNNX78eF28eNE+r8lk0qxZs3TXXXepevXqmjhxos6cOaO+ffsqODhYPj4+ioqKUkpKiqTLhyVMJpO2bdtmX8auXbt05513yt/fX35+furUqZMOHDggSSooKNCECRNUv359eXl5qWXLllq6dOlVv681a9aoffv28vLyUmhoqP7xj38Uyty1a1cNGjRIQ4cOVVBQkHr06FGq7VYalAkAQKX2wAMP6NSpU1q1apV92unTp7V06VL17dtX69at0yOPPKIhQ4Zo9+7deuedd5SamqqJEycWWs64ceN077336qefftJjjz2mMWPGaPfu3frmm29kNps1a9asYg9rHD16VJ07d5aXl5dWrlyptLQ0PfbYY/Y3/xkzZmjatGmaOnWqduzYoR49euiuu+7Svn1F32796NGjuuOOO9SuXTtt375ds2bN0nvvvadXX3210Hzvv/++PD09tX79es2ePdvIZrwqDnMAACq1G264QfHx8Zo/f766desmSVq4cKGCgoJ06623qnv37vrHP/6h/v37S5IaNmyoV155RSNHjtTYsWPty3n44Yf16KOP2p9bLBa1atVKbdu2lSQ1aNCg2AzJyckKCAjQggUL7IdHmjRpYn996tSpGjVqlPr06SNJmjJlilatWqXp06crOTn5T8t7++23FRYWpn/+858ymUy68cYbdezYMY0aNUqJiYmqUuXyvoKoqCi99tprjmy2UmHPBACg0uvbt68+++wz5eXlSZLmzZunPn36qEqVKtq+fbsmTJigGjVq2B9PPvmkMjIyCp1UeqU0XPHss89qwYIFatmypUaOHKnvv/++2PVv27ZNnTp1KvI8i6ysLB07dkxxcXGFpsfFxclsNhe5PLPZrI4dO8pkMhWaPzs7W7/88ot9Wps2ba6yVcoOZQIAUOn16tVLNptNX331lY4cOaJ169apb9++kqTs7GyNHz9e27Ztsz9++ukn7du3T97e3vZlVK9evdAy4+PjdfjwYb3wwgs6duyYunXrpuHDhxe5fh8fH+d9c1fxx8zOQpkAAFR63t7e6t27t+bNm6ePPvpI0dHRat26tSSpdevW2rt3rxo3bvynx5XDBcUJDg5W//799eGHH2r69OmaM2dOkfM1b95c69at04ULF/70mr+/v+rWrav169cXmr5+/XrFxsYWubyYmBht2LBBNput0Px+fn6qX7/+VTM7A+dMAACuC3379tWdd96pXbt2qV+/fvbpiYmJuvPOOxUeHq7777/ffuhj586dfzqh8fcSExPVpk0bNW3aVHl5eVqyZIliYmKKnHfQoEGaOXOm+vTpo4SEBAUEBGjjxo1q3769oqOjNWLECI0dO1aNGjVSy5YtlZKSom3btmnevHlFLu+5557T9OnTNXjwYA0aNEh79+7V2LFjNWzYsGsWIGegTAAAyoT54BG3Xs9tt92mwMBA7d27Vw8//LB9eo8ePbRkyRJNmDBBU6ZMUbVq1XTjjTfqiSeeuOryPD09lZCQoEOHDsnHx0edOnXSggULipy3Vq1aWrlypUaMGKEuXbrIw8NDLVu2tJ8n8fzzzyszM1MvvviiTp48qdjYWC1evFhRUVFFLq9evXr6+uuvNWLECLVo0UKBgYF6/PHHNXr0aIe2jVEm2+/3kVRCWVlZCggIUGZmpvz9/V0dxyFbt25VmzZt9Pan3ykqtnmJx+3bvUPPPdBdaWlp9t15AOCo8+fPKz09XZGRkYXOJeAKmBVbcT9XqeTvoeyZAAAYEh4eLrPZzL05rmOUCQCAYeHh4by5X8f4NAcAADCEMgEAAAyhTAAAAEMoEwAAwBDKBAAAMIQyAQAADKFMAAAAQ7jOBADAMIvFUmEvWjVu3DgtWrRI27ZtM7Sc1atX69Zbb9WZM2dUs2bNEo0ZMGCAzp49q0WLFhlat6tRJgAAhlT0y2kPHz5cgwcPNrycm2++WRkZGQoICCjxmBkzZqgy3NWCMgEAMMRqtSo3N1cvzXhJEY0jnL6+w/sPa9KQSbJarWVSJmrUqKEaNWoU+3p+fr48PT2vuRxPT0+FhISUat2lKR7ujDIBACgTEY0j1KRZE1fH+JM5c+Zo3Lhx+uWXXwrdnvvuu+9WrVq1FB4eXugwx5VDD+3atVNycrK8vLyUnp6u77//Xs8995z27Nmjm266SaNHj9a9996rH3/8US1btvzTYY7U1FQNHTpUH3/8sYYOHaojR47olltuUUpKikJDQwut68phjoKCAk2dOlVz5szRkSNHVKdOHT399NN6+eWXJUmjRo3SF198oV9++UUhISHq27evEhMTVa1atXLdpn/ECZgAgErtgQce0KlTp7Rq1Sr7tNOnT2vp0qXq27dvkWNWrFihvXv3atmyZVqyZImysrLUq1cvNWvWTFu3btUrr7yiUaNGXXPdubm5mjp1qv79739r7dq1slgsGj58eLHzJyQkaPLkyRozZox2796t+fPnq06dOvbX/fz8lJqaqt27d2vGjBl699139eabb5ZiazgHeyYAAJXaDTfcoPj4eM2fP1/dunWTJC1cuFBBQUG69dZbtW7duj+NqV69uv71r3/ZD2/Mnj1bJpNJ7777rry9vRUbG6ujR4/qySefvOq6L1y4oNmzZ6tRo0aSpEGDBmnChAlFznvu3DnNmDFD//znP9W/f39JUqNGjXTLLbfY5xk9erT96wYNGmj48OFasGCBRo4cWYotUvbYMwEAqPT69u2rzz77THl5eZKkefPmqU+fPoUOe/xes2bNCp0nsXfvXjVv3lze3t72ae3bt7/men19fe1FQpJCQ0N18uTJIuc1m83Ky8uzF56ifPzxx4qLi1NISIhq1Kih0aNHy2KxXDOHs1EmAACVXq9evWSz2fTVV1/pyJEjWrduXbGHOKTLeybKwh/PZTCZTMV+esPHx+eqy9qwYYP69u2rO+64Q0uWLNGPP/6ol19+Wfn5+WWS1QjKBACg0vP29lbv3r01b948ffTRR4qOjlbr1q1LPD46Olo//fSTfc+GJG3evLlMM0ZFRcnHx0crVqwo8vXvv/9eERERevnll9W2bVtFRUXp8OHDZZrBUZQJAMB1oW/fvvrqq680d+7cq+6VKMrDDz+sgoICPfXUUzKbzfr22281depUSZf3NpQFb29vjRo1SiNHjtQHH3ygAwcOaOPGjXrvvfckXS4bFotFCxYs0IEDB/TWW2/piy++KJN1G8UJmACAMnF4f/n8L9nR9dx2220KDAzU3r179fDDD5dqrL+/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PDycG0YBJeG0u34AcKkuXbrYatSoYRsyZIhtz549tg8//NDm6+trvwnSE088Ybv55ptta9eute3fv9/2+uuv27y8vGw///yzLS8vzzZ9+nSbv7+/LSMjw5aRkWE7d+6czWaz2d577z3b119/bTtw4IBtw4YNto4dO9ri4+Pt671yQ7wzZ85cM2NKSootICDA/nzt2rU2f39/W2pqqu3AgQO27777ztagQQPbuHHj7PNIstWvX982f/582759+2zPP/+8rUaNGrZTp07ZbDab7eDBg7Zq1arZhg8fbtuzZ4/to48+stWrV8+e6WrfW0REhC0wMNCWnJxs27dvny0pKclWpUoV2549e4z+OIBKjTIBVFJdunSxxcTE2AoKCuzTRo0aZYuJibEdPnzY5uHhYTt69GihMd26dbMlJCTYbLY/v9EXZ/PmzTZJ9jdkI2WiW7dutkmTJhWa59///rctNDTU/lySbfTo0fbn2dnZNkm2b775xv493nTTTYWW8fLLLxfKVNz3FhERYevXr5/9eUFBga127dq2WbNmXfN7Aa5nTrqxOQB38Je//EUmk8n+vGPHjpo2bZp++uknXbp0SU2aNCk0f15enmrVqnXVZaalpWncuHHavn27zpw5o4KCAkmSxWJRbGysobzbt2/X+vXrNXHiRPu0S5cu6fz588rNzZWvr68kqXnz5vbXq1evLn9/f/uhlr1796pdu3aFltu+ffsSZ/j9sk0mk0JCQsr0MA5QGVEmgOtQdna2PDw8lJaWJg8Pj0Kv1ahRo9hxOTk56tGjh3r06KF58+YpODhYFotFPXr0UH5+fpnkGj9+vHr37v2n17y9ve1fV6tWrdBrJpPJXmqMcuaygcqKMgFUYps2bSr0fOPGjYqKilKrVq106dIlnTx5Up06dSpyrKenpy5dulRo2p49e3Tq1ClNnjxZYWFhkqQtW7aUWd7WrVtr7969aty4scPLiI6O1tdff11o2ubNmws9L+p7A+A4Ps0BVGIWi0XDhg3T3r179dFHH2nmzJkaMmSImjRpor59++qRRx7R559/rvT0dP3www9KSkrSV199JenyBZyys7O1YsUKWa1W5ebmKjw8XJ6enpo5c6YOHjyoxYsX65VXXimzvImJifrggw80fvx47dq1S2azWQsWLNDo0aNLvIynn35ae/bs0ahRo/Tzzz/rk08+UWpqqiTZD/kU9b0BcBxlAqjEHnnkEf32229q3769Bg4cqCFDhuipp56SJKWkpOiRRx7Riy++qOjoaN1zzz3avHmzwsPDJUk333yznnnmGT344IMKDg7Wa6+9puDgYKWmpurTTz9VbGysJk+erKlTp5ZZ3h49emjJkiX67rvv1K5dO/3lL3/Rm2++qYiIiBIvIzIyUgsXLtTnn3+u5s2ba9asWXr55ZclSV5eXsV+bwAcZ7LZ/vehcwCVSteuXdWyZUsuDy1p4sSJmj17to4cOeLqKEClxDkTACqdt99+W+3atVOtWrW0fv16vf766xo0aJCrYwGVFoc5ADhNfHy8atSoUeRj0qRJTlvvvn37dPfddys2NlavvPKKXnzxxUJXuQRQtjjMAcBpjh49qt9++63I1wIDAxUYGFjOiQA4A2UCAAAYwmEOAABgCGUCAAAYQpkAAACGUCYAAIAhlAkAAGAIZQIAABhCmQAAAIZQJgAAgCH/D10C0mAvzHbhAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "hist_graphing('petal_length', 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhMAAAINCAYAAACEf/3PAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABBSklEQVR4nO3dfVxUZf7/8feAAqKCISiQgKBIaCretmompom0Wma1mbppN/atxDLzJirvK9I0NSOt3dJuNLdstTJXV1HUvKtQ8o78eT+meEMqiCigzO8P19klQZk5wAzwej4e5/Fgzrmucz5cjDNvz5w5l8lisVgEAABgJxdHFwAAACo2wgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQ6o5uoCyVlBQoOPHj6t27doymUyOLgcAgArDYrHo/PnzCgwMlItL8ecfKn2YOH78uIKCghxdBgAAFdbRo0fVoEGDYrdX+jBRu3ZtSVcHwsvLy8HVAABQcWRlZSkoKMj6Xloch4aJOXPmaM6cOTp8+LAkqVmzZho3bpxiY2MlSZcuXdJLL72kRYsWKTc3VzExMXr//fdVv379Eh/j2kcbXl5ehAkAAOxws8sEHHoBZoMGDfTWW28pJSVFP//8s+6++27df//92r17tyTpxRdf1HfffaevvvpK69at0/Hjx9W3b19HlgwAAP7A5Gyzhvr4+Ojtt9/WQw89JD8/Py1cuFAPPfSQJOnXX39VZGSkNm/erD/96U8l2l9WVpa8vb2VmZnJmQkAAGxQ0vdQp/lq6JUrV7Ro0SJduHBBHTp0UEpKivLz89W9e3drm9tuu03BwcHavHlzsfvJzc1VVlZWoQUAAJQdh1+AuXPnTnXo0EGXLl1SrVq1tGTJEjVt2lSpqalyc3NTnTp1CrWvX7++Tpw4Uez+EhISNHHixDKuGgCqLovFosuXL+vKlSuOLgUGubq6qlq1aoZvneDwMBEREaHU1FRlZmZq8eLFGjRokNatW2f3/uLj4zVixAjr42tXogIAjMvLy1N6erpycnIcXQpKiaenpwICAuTm5mb3PhweJtzc3NS4cWNJUps2bfTTTz9p1qxZeuSRR5SXl6dz584VOjtx8uRJ+fv7F7s/d3d3ubu7l3XZAFDlFBQU6NChQ3J1dVVgYKDc3Ny4GWAFZrFYlJeXp9OnT+vQoUMKDw+/4Y2pbsThYeKPCgoKlJubqzZt2qh69epKSkrSgw8+KEnau3evzGazOnTo4OAqAaDqycvLU0FBgYKCguTp6enoclAKatSooerVq+vIkSPKy8uTh4eHXftxaJiIj49XbGysgoODdf78eS1cuFDJyclauXKlvL299eSTT2rEiBHy8fGRl5eXhg0bpg4dOpT4mxwAgNJn7/9e4ZxK4+/p0DBx6tQpPfbYY0pPT5e3t7datGihlStX6p577pEkzZgxQy4uLnrwwQcL3bQKAAA4D6e7z0Rp4z4TAFA6Ll26pEOHDik0NNTu0+GVxeDBg3Xu3DktXbrU0aUYdqO/a0nfQ53umgkAAJzdrFmzVMn/L24TwgQAADby9vZ2dAlOhatoAAAV0uLFi9W8eXPVqFFDdevWVffu3XXhwgUNHjxYffr00cSJE+Xn5ycvLy8988wzysvLs/YtKChQQkKCQkNDVaNGDbVs2VKLFy8utP/du3erV69e8vLyUu3atdW5c2cdOHBAkqzHKOn+zp49qwEDBsjPz081atRQeHi45s2bV7YDVI44MwEAqHDS09P16KOPaurUqXrggQd0/vx5bdiwwfrRQ1JSkjw8PJScnKzDhw/r8ccfV926dfXGG29Iunq35M8//1xz585VeHi41q9fr4EDB8rPz09dunTRsWPHdNdddyk6Olpr1qyRl5eXNm7cqMuXLxdZz832N3bsWO3Zs0f/+te/5Ovrq/379+vixYvlNl5ljTABAKhw0tPTdfnyZfXt21chISGSpObNm1u3u7m56eOPP5anp6eaNWumSZMmadSoUZo8ebLy8/P15ptvavXq1db7FoWFhemHH37QBx98oC5duigxMVHe3t5atGiRqlevLklq0qRJkbXk5ubedH9ms1mtWrVS27ZtJUkNGzYsq6FxCMIEAKDCadmypbp166bmzZsrJiZGPXr00EMPPaRbbrnFuv1/b6zVoUMHZWdn6+jRo8rOzlZOTo71NgTX5OXlqVWrVpKk1NRUde7c2RokbmT//v033d+zzz6rBx98UNu2bVOPHj3Up08fdezY0dAYOBPCBACgwnF1ddWqVau0adMm/fvf/9bs2bP16quvauvWrTftm52dLUn6/vvvdeuttxbadm06hho1apS4lpLsLzY2VkeOHNHy5cu1atUqdevWTUOHDtW0adNKfBxnRpgAAFRIJpNJnTp1UqdOnTRu3DiFhIRoyZIlkqRffvlFFy9etIaCLVu2qFatWgoKCpKPj4/c3d1lNpvVpUuXIvfdokULffLJJ8rPz7/p2YmmTZvedH+S5Ofnp0GDBmnQoEHq3LmzRo0aRZiAfcxmszIyMmzu5+vrq+Dg4DKoCAAqnq1btyopKUk9evRQvXr1tHXrVp0+fVqRkZHasWOH8vLy9OSTT+q1117T4cOHNX78eMXFxcnFxUW1a9fWyJEj9eKLL6qgoEB33nmnMjMztXHjRnl5eWnQoEGKi4vT7Nmz1a9fP8XHx8vb21tbtmxR+/btFRERUaiWkuxv3LhxatOmjZo1a6bc3FwtW7ZMkZGRDhq90keYKEdms1mRkZF2Td3r6emptLQ0AgUASPLy8tL69es1c+ZMZWVlKSQkRNOnT1dsbKz+8Y9/qFu3bgoPD9ddd92l3NxcPfroo5owYYK1/+TJk+Xn56eEhAQdPHhQderUUevWrfXKK69IkurWras1a9Zo1KhR6tKli1xdXRUVFaVOnToVWc/N9ufm5qb4+HgdPnxYNWrUUOfOnbVo0aIyH6fywu20y9G2bdvUpk0bvTwlUcFh4SXuZz64T2+NGaqUlBS1bt26DCsEgOJVlNtpV6ZbXZcHbqddQQWHhSu8aQtHlwEAQKngDpgAAMAQzkwAACqV+fPnO7qEKoczEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhK+GAgAMs3feIXsxX5FzIUwAAAwxMu+QvcprvqLDhw8rNDRU27dvV1RUVJkeqyIjTAAADMnIyFBOTo7N8w7Z69p8RRkZGZydcBKECQBAqXDmeYcWL16siRMnav/+/fL09FSrVq30zTffqGbNmvr73/+u6dOn69ChQ2rYsKGef/55Pffcc5Kk0NBQSVKrVq0kSV26dFFycrIKCgr0+uuv68MPP7ROff7WW2+pZ8+ekqS8vDyNGDFCX3/9tc6ePav69evrmWeeUXx8vCTpnXfe0bx583Tw4EH5+Piod+/emjp1qmrVquWA0TGOMAEAqNTS09P16KOPaurUqXrggQd0/vx5bdiwQRaLRQsWLNC4ceP03nvvqVWrVtq+fbuGDBmimjVratCgQfrxxx/Vvn17rV69Ws2aNZObm5skadasWZo+fbo++OADtWrVSh9//LHuu+8+7d69W+Hh4Xr33Xf17bff6ssvv1RwcLCOHj2qo0ePWmtycXHRu+++q9DQUB08eFDPPfecRo8erffff99Rw2QIYQIAUKmlp6fr8uXL6tu3r0JCQiRJzZs3lySNHz9e06dPV9++fSVdPROxZ88effDBBxo0aJD8/PwkSXXr1pW/v791n9OmTdOYMWPUr18/SdKUKVO0du1azZw5U4mJiTKbzQoPD9edd94pk8lkPe41w4cPt/7csGFDvf7663rmmWcIEwAAOKOWLVuqW7duat68uWJiYtSjRw899NBDcnNz04EDB/Tkk09qyJAh1vaXL1+Wt7d3sfvLysrS8ePH1alTp0LrO3XqpF9++UWSNHjwYN1zzz2KiIhQz5491atXL/Xo0cPadvXq1UpISNCvv/6qrKwsXb58WZcuXVJOTo48PT1LeQTKHveZAABUaq6urlq1apX+9a9/qWnTppo9e7YiIiK0a9cuSdLf/vY3paamWpddu3Zpy5Ytho7ZunVrHTp0SJMnT9bFixf1l7/8RQ899JCkq98Q6dWrl1q0aKGvv/5aKSkpSkxMlHT1WouKiDMTAIBKz2QyqVOnTurUqZPGjRunkJAQbdy4UYGBgTp48KAGDBhQZL9r10hcuXLFus7Ly0uBgYHauHGjunTpYl2/ceNGtW/fvlC7Rx55RI888ogeeugh9ezZU2fOnFFKSooKCgo0ffp0ubhc/T/9l19+WRa/drkhTAAAKrWtW7cqKSlJPXr0UL169bR161brNzAmTpyo559/Xt7e3urZs6dyc3P1888/6+zZsxoxYoTq1aunGjVqaMWKFWrQoIE8PDzk7e2tUaNGafz48WrUqJGioqI0b948paamasGCBZKuflsjICBArVq1kouLi7766iv5+/urTp06aty4sfLz8zV79mz17t1bGzdu1Ny5cx08SsYQJgAApcJ8cJ9THsfLy0vr16/XzJkzlZWVpZCQEE2fPl2xsbGSrt4A6+2339aoUaNUs2ZNNW/e3HqBZLVq1fTuu+9q0qRJGjdunDp37qzk5GQ9//zzyszM1EsvvaRTp06padOm+vbbbxUefvU+G7Vr19bUqVO1b98+ubq6ql27dlq+fLlcXFzUsmVLvfPOO5oyZYri4+N11113KSEhQY899lipjlN5MlksFoujiyhLWVlZ8vb2VmZmpry8vBxay7Zt29SmTRu9/9W/bfou9r49O/Tcwz2UkpKi1q1bl2GFAFC8S5cu6dChQwoNDZWHh4d1fWW+A2ZVUNzfVSr5eyhnJgAAhgQHBystLY25OaowwgQAwLDg4GDe3KswvhoKAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDuMwEAMMxsNnPTqv9x+PBhhYaGavv27YqKinK6/ZU2wgQAwBBup329oKAgpaeny9fX19GllAvCBADAkIyMDOXk5Ojzt0cpMiyozI+XdvCoBo56WxkZGQ4LE/n5+apevXqx211dXeXv71+OFd1cXl6edUr10sY1EwCAUhEZFqTWzRqX+WJrYPnwww8VGBiogoKCQuvvv/9+PfHEE5Kkb775Rq1bt5aHh4fCwsI0ceJEXb582drWZDJpzpw5uu+++1SzZk298cYbOnv2rAYMGCA/Pz/VqFFD4eHhmjdvnqSrH0uYTCalpqZa97F792716tVLXl5eql27tjp37qwDBw5IkgoKCjRp0iQ1aNBA7u7uioqK0ooVK274e61bt07t27eXu7u7AgIC9PLLLxeqOTo6WnFxcRo+fLh8fX0VExNj07jZgjABAKjUHn74Yf3+++9au3atdd2ZM2e0YsUKDRgwQBs2bNBjjz2mF154QXv27NEHH3yg+fPn64033ii0nwkTJuiBBx7Qzp079cQTT2js2LHas2eP/vWvfyktLU1z5swp9mONY8eO6a677pK7u7vWrFmjlJQUPfHEE9Y3/1mzZmn69OmaNm2aduzYoZiYGN13333at6/o6daPHTume++9V+3atdMvv/yiOXPm6KOPPtLrr79eqN0nn3wiNzc3bdy4UXPnzjUyjDfExxwAgErtlltuUWxsrBYuXKhu3bpJkhYvXixfX1917dpVPXr00Msvv6xBgwZJksLCwjR58mSNHj1a48ePt+6nf//+evzxx62PzWazWrVqpbZt20qSGjZsWGwNiYmJ8vb21qJFi6wfjzRp0sS6fdq0aRozZoz69esnSZoyZYrWrl2rmTNnKjEx8br9vf/++woKCtJ7770nk8mk2267TcePH9eYMWM0btw4ubhcPVcQHh6uqVOn2jNsNuHMBACg0hswYIC+/vpr5ebmSpIWLFigfv36ycXFRb/88osmTZqkWrVqWZchQ4YoPT290EWl10LDNc8++6wWLVqkqKgojR49Wps2bSr2+KmpqercuXOR11lkZWXp+PHj6tSpU6H1nTp1UlpaWpH7S0tLU4cOHWQymQq1z87O1m+//WZd16ZNmxuMSukhTAAAKr3evXvLYrHo+++/19GjR7VhwwYNGDBAkpSdna2JEycqNTXVuuzcuVP79u2Th4eHdR81a9YstM/Y2FgdOXJEL774oo4fP65u3bpp5MiRRR6/Ro0aZffL3cAfay4rhAkAQKXn4eGhvn37asGCBfriiy8UERGh1q1bS5Jat26tvXv3qnHjxtct1z4uKI6fn58GDRqkzz//XDNnztSHH35YZLsWLVpow4YNys/Pv26bl5eXAgMDtXHjxkLrN27cqKZNmxa5v8jISG3evFkWi6VQ+9q1a6tBgwY3rLkscM0EAKBKGDBggHr16qXdu3dr4MCB1vXjxo1Tr169FBwcrIceesj60ceuXbuuu6Dxf40bN05t2rRRs2bNlJubq2XLlikyMrLItnFxcZo9e7b69eun+Ph4eXt7a8uWLWrfvr0iIiI0atQojR8/Xo0aNVJUVJTmzZun1NRULViwoMj9Pffcc5o5c6aGDRumuLg47d27V+PHj9eIESNuGoDKAmECAFAq0g4ederj3H333fLx8dHevXvVv39/6/qYmBgtW7ZMkyZN0pQpU1S9enXddttteuqpp264Pzc3N8XHx+vw4cOqUaOGOnfurEWLFhXZtm7dulqzZo1GjRqlLl26yNXVVVFRUdbrJJ5//nllZmbqpZde0qlTp9S0aVN9++23Cg8PL3J/t956q5YvX65Ro0apZcuW8vHx0ZNPPqnXXnvNrrExymT533MklVBWVpa8vb2VmZkpLy8vh9aybds2tWnTRu9/9W+FN21R4n779uzQcw/3UEpKivW0HACUt0uXLunQoUMKDQ0tdC0Bd8Cs2Ir7u0olfw/lzAQAwJDg4GClpaUxN0cVRpgAABgWHBzMm3sVxrc5AACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABjCfSYAAIaZzeYKe9OqCRMmaOnSpUpNTTW0n+TkZHXt2lVnz55VnTp1StRn8ODBOnfunJYuXWro2I5GmAAAGFLRb6c9cuRIDRs2zPB+OnbsqPT0dHl7e5e4z6xZs1QZZrUgTAAADMnIyFBOTo5emfWKQhqHlPnxjuw/ojdfeFMZGRmlEiZq1aqlWrVqFbs9Ly9Pbm5uN92Pm5ub/P39bTq2LcHDmTn0momEhAS1a9dOtWvXVr169dSnTx/t3bu3UJvo6GiZTKZCyzPPPOOgigEAxQlpHKImzZuU+WJrYPnwww8VGBiogoKCQuvvv/9+PfHEE5owYYKioqKs6wcPHqw+ffrojTfeUGBgoCIiIiRJmzZtUlRUlDw8PNS2bVstXbpUJpPJ+vFIcnKyTCaTzp07J0maP3++6tSpo5UrVyoyMlK1atVSz549lZ6eft2xrikoKNDUqVPVuHFjubu7Kzg4WG+88YZ1+5gxY9SkSRN5enoqLCxMY8eOVX5+vk3jURYcGibWrVunoUOHasuWLVq1apXy8/PVo0cPXbhwoVC7IUOGKD093bpMnTrVQRUDACqahx9+WL///rvWrl1rXXfmzBmtWLFCAwYMKLJPUlKS9u7dq1WrVmnZsmXKyspS79691bx5c23btk2TJ0/WmDFjbnrsnJwcTZs2TZ999pnWr18vs9mskSNHFts+Pj5eb731lsaOHas9e/Zo4cKFql+/vnV77dq1NX/+fO3Zs0ezZs3S3/72N82YMcOG0SgbDv2YY8WKFYUez58/X/Xq1VNKSoruuusu63pPT0+bTx0BACBJt9xyi2JjY7Vw4UJ169ZNkrR48WL5+vqqa9eu2rBhw3V9atasqb///e/Wjzfmzp0rk8mkv/3tb/Lw8FDTpk117NgxDRky5IbHzs/P19y5c9WoUSNJUlxcnCZNmlRk2/Pnz2vWrFl67733NGjQIElSo0aNdOedd1rbvPbaa9afGzZsqJEjR2rRokUaPXq0DSNS+pzqq6GZmZmSJB8fn0LrFyxYIF9fX91+++2Kj4+/4UU+ubm5ysrKKrQAAKq2AQMG6Ouvv1Zubq6kq+8r/fr1k4tL0W+DzZs3L3SdxN69e9WiRQt5eHhY17Vv3/6mx/X09LQGCUkKCAjQqVOnimyblpam3Nxca+Apyj/+8Q916tRJ/v7+qlWrll577TWZzeab1lHWnCZMFBQUaPjw4erUqZNuv/126/r+/fvr888/19q1axUfH6/PPvtMAwcOLHY/CQkJ8vb2ti5BQUHlUT4AwIn17t1bFotF33//vY4ePaoNGzYU+xGHdPXMRGmoXr16occmk6nYb2/UqFHjhvvavHmzBgwYoHvvvVfLli3T9u3b9eqrryovL69UajXCab7NMXToUO3atUs//PBDofVPP/209efmzZsrICBA3bp104EDBwqlvWvi4+M1YsQI6+OsrCwCBQBUcR4eHurbt68WLFig/fv3KyIiQq1bty5x/4iICH3++efKzc2Vu7u7JOmnn34q1RrDw8NVo0YNJSUl6amnnrpu+6ZNmxQSEqJXX33Vuu7IkSOlWoO9nOLMRFxcnJYtW6a1a9eqQYMGN2x7xx13SJL2799f5HZ3d3d5eXkVWgAAGDBggL7//nt9/PHHNzwrUZT+/furoKBATz/9tNLS0rRy5UpNmzZN0tWzDaXBw8NDY8aM0ejRo/Xpp5/qwIED2rJliz766CNJV8OG2WzWokWLdODAAb377rtasmRJqRzbKIeembBYLBo2bJiWLFmi5ORkhYaG3rTPta/gBAQElHF1AABbHNlfPv9Ltvc4d999t3x8fLR3717179/fpr5eXl767rvv9OyzzyoqKkrNmzfXuHHj1L9//0LXURg1duxYVatWTePGjdPx48cVEBBgvR3CfffdpxdffFFxcXHKzc3Vn//8Z40dO1YTJkwotePby2Rx4K23nnvuOS1cuFDffPON9Xu80tWbeNSoUUMHDhzQwoULde+996pu3brasWOHXnzxRTVo0EDr1q0r0TGysrLk7e2tzMxMh5+l2LZtm9q0aaP3v/q3wpu2KHG/fXt26LmHeyglJcWm03IAUJouXbqkQ4cOKTQ0tNAbaEW/A6a9FixYoMcff1yZmZk3vd7BmRX3d5VK/h7q0DMTc+bMkXT1xlT/a968eRo8eLDc3Ny0evVqzZw5UxcuXFBQUJAefPDBQl+NAQA4VnBwsNLS0irs3Bwl9emnnyosLEy33nqrfvnlF40ZM0Z/+ctfKnSQKC0O/5jjRoKCgkp8BgIA4DjBwcEOPUtQHk6cOKFx48bpxIkTCggI0MMPP1zo7pRVmdN8mwMAAGc2evRoh98cylk5xbc5AABAxUWYAAAAhhAmAAA2+ePsm6jYSuPvyTUTAIAScXNzk4uLi44fPy4/Pz+5ubmV2g2bUP4sFovy8vJ0+vRpubi4FJqLxFaECQBAibi4uCg0NFTp6ek6fvy4o8tBKfH09FRwcHCxk56VBGECAFBibm5uCg4O1uXLl3XlyhVHlwODXF1dVa1aNcNnmAgTAACbmEwmVa9e/boZMVF1cQEmAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMMShYSIhIUHt2rVT7dq1Va9ePfXp00d79+4t1ObSpUsaOnSo6tatq1q1aunBBx/UyZMnHVQxAAD4I4eGiXXr1mno0KHasmWLVq1apfz8fPXo0UMXLlywtnnxxRf13Xff6auvvtK6det0/Phx9e3b14FVAwCA/1XNkQdfsWJFocfz589XvXr1lJKSorvuukuZmZn66KOPtHDhQt19992SpHnz5ikyMlJbtmzRn/70J0eUDQAA/odTXTORmZkpSfLx8ZEkpaSkKD8/X927d7e2ue222xQcHKzNmzcXuY/c3FxlZWUVWgAAQNlxmjBRUFCg4cOHq1OnTrr99tslSSdOnJCbm5vq1KlTqG39+vV14sSJIveTkJAgb29v6xIUFFTWpQMAUKU5TZgYOnSodu3apUWLFhnaT3x8vDIzM63L0aNHS6lCAABQFIdeM3FNXFycli1bpvXr16tBgwbW9f7+/srLy9O5c+cKnZ04efKk/P39i9yXu7u73N3dy7pkAADwHw49M2GxWBQXF6clS5ZozZo1Cg0NLbS9TZs2ql69upKSkqzr9u7dK7PZrA4dOpR3uQAAoAgOPTMxdOhQLVy4UN98841q165tvQ7C29tbNWrUkLe3t5588kmNGDFCPj4+8vLy0rBhw9ShQwe+yQEAgJNwaJiYM2eOJCk6OrrQ+nnz5mnw4MGSpBkzZsjFxUUPPvigcnNzFRMTo/fff7+cKwUAAMVxaJiwWCw3bePh4aHExEQlJiaWQ0UAAMBWTvNtDgAAUDERJgAAgCGECQAAYAhhAgAAGEKYAAAAhhAmAACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYAhhAgAAGEKYAAAAhhAmAACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYAhhAgAAGEKYAAAAhhAmAACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYAhhAgAAGEKYAAAAhhAmAACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhiV5gICwvT77//ft36c+fOKSwszHBRAACg4rArTBw+fFhXrly5bn1ubq6OHTtmuCgAAFBxVLOl8bfffmv9eeXKlfL29rY+vnLlipKSktSwYcNSKw4AADg/m8JEnz59JEkmk0mDBg0qtK169epq2LChpk+fXmrFAQAA52dTmCgoKJAkhYaG6qeffpKvr2+ZFAUAACoOm8LENYcOHSrtOgAAQAVlV5iQpKSkJCUlJenUqVPWMxbXfPzxx4YLAwAAFYNdYWLixImaNGmS2rZtq4CAAJlMptKuCwAAVBB2hYm5c+dq/vz5+utf/1ra9QAAgArGrvtM5OXlqWPHjqVdCwAAqIDsChNPPfWUFi5cWNq1AACACsiujzkuXbqkDz/8UKtXr1aLFi1UvXr1QtvfeeedUikOAAA4P7vCxI4dOxQVFSVJ2rVrV6FtXIwJAEDVYleYWLt2bWnXAQAAKiimIAcAAIbYdWaia9euN/w4Y82aNXYXBAAAKha7wsS16yWuyc/PV2pqqnbt2nXdBGAAAKBysytMzJgxo8j1EyZMUHZ2don3s379er399ttKSUlRenq6lixZYp2ZVJIGDx6sTz75pFCfmJgYrVixwp6yAQBAGSjVayYGDhxo07wcFy5cUMuWLZWYmFhsm549eyo9Pd26fPHFF6VRKgAAKCV2T/RVlM2bN8vDw6PE7WNjYxUbG3vDNu7u7vL39zdaGgAAKCN2hYm+ffsWemyxWJSenq6ff/5ZY8eOLZXCrklOTla9evV0yy236O6779brr7+uunXrFts+NzdXubm51sdZWVmlWg8AACjMrjDh7e1d6LGLi4siIiI0adIk9ejRo1QKk65+xNG3b1+FhobqwIEDeuWVVxQbG6vNmzfL1dW1yD4JCQmaOHFiqdUAAABuzK4wMW/evNKuo0j9+vWz/ty8eXO1aNFCjRo1UnJysrp161Zkn/j4eI0YMcL6OCsrS0FBQWVeKwAAVZWhayZSUlKUlpYmSWrWrJlatWpVKkUVJywsTL6+vtq/f3+xYcLd3V3u7u5lWgcAAPgvu8LEqVOn1K9fPyUnJ6tOnTqSpHPnzqlr165atGiR/Pz8SrNGq99++02///67AgICymT/AADAdnZ9NXTYsGE6f/68du/erTNnzujMmTPatWuXsrKy9Pzzz5d4P9nZ2UpNTVVqaqok6dChQ0pNTZXZbFZ2drZGjRqlLVu26PDhw0pKStL999+vxo0bKyYmxp6yAQBAGbDrzMSKFSu0evVqRUZGWtc1bdpUiYmJNl2A+fPPP6tr167Wx9eudRg0aJDmzJmjHTt26JNPPtG5c+cUGBioHj16aPLkyXyMAQCAE7ErTBQUFKh69erXra9evboKCgpKvJ/o6GhZLJZit69cudKe8gAAQDmy62OOu+++Wy+88IKOHz9uXXfs2DG9+OKLxV4YCQAAKie7wsR7772nrKwsNWzYUI0aNVKjRo0UGhqqrKwszZ49u7RrBAAATsyujzmCgoK0bds2rV69Wr/++qskKTIyUt27dy/V4gAAgPOz6czEmjVr1LRpU2VlZclkMumee+7RsGHDNGzYMLVr107NmjXThg0byqpWAADghGwKEzNnztSQIUPk5eV13TZvb2/93//9n955551SKw4AADg/m8LEL7/8op49exa7vUePHkpJSTFcFAAAqDhsChMnT54s8iuh11SrVk2nT582XBQAAKg4bAoTt956q3bt2lXs9h07dnCrawAAqhibwsS9996rsWPH6tKlS9dtu3jxosaPH69evXqVWnEAAMD52fTV0Ndee03//Oc/1aRJE8XFxSkiIkKS9OuvvyoxMVFXrlzRq6++WiaFAgAA52RTmKhfv742bdqkZ599VvHx8dZbYZtMJsXExCgxMVH169cvk0IBAIBzsvmmVSEhIVq+fLnOnj2r/fv3y2KxKDw8XLfccktZ1AcAAJycXXfAlKRbbrlF7dq1K81aAABABWTX3BwAAADXECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYAhhAgAAGEKYAAAAhth9B0wAKA1ms1kZGRk29/P19VVwcHAZVATAVoQJAA5jNpsVGRmpnJwcm/t6enoqLS2NQAE4AcIEAIfJyMhQTk6OPn97lCLDgkrcL+3gUQ0c9bYyMjIIE4ATIEwAcLjIsCC1btbY0WUAsBMXYAIAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMKSaowsAAJQus9msjIwMm/v5+voqODi4DCpCZUeYAIBKxGw2KzIyUjk5OTb39fT0VFpaGoECNiNMAEAlkpGRoZycHL0y6xWFNA4pcb8j+4/ozRfeVEZGBmECNiNMAEAlFNI4RE2aN3F0GagiuAATAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYIhDw8T69evVu3dvBQYGymQyaenSpYW2WywWjRs3TgEBAapRo4a6d++uffv2OaZYAABQJIeGiQsXLqhly5ZKTEwscvvUqVP17rvvau7cudq6datq1qypmJgYXbp0qZwrBQAAxXHo7bRjY2MVGxtb5DaLxaKZM2fqtdde0/333y9J+vTTT1W/fn0tXbpU/fr1K89SAQBAMZx2bo5Dhw7pxIkT6t69u3Wdt7e37rjjDm3evLnYMJGbm6vc3Fzr46ysrDKvFUDFwhTdQOly2jBx4sQJSVL9+vULra9fv751W1ESEhI0ceLEMq0NQMXFFN1A6XPaMGGv+Ph4jRgxwvo4KytLQUFBDqwIgDO5NkX352+PUmRYyV8b0g4e1cBRbzNFN1AEpw0T/v7+kqSTJ08qICDAuv7kyZOKiooqtp+7u7vc3d3LujwAFVxkWJBaN2vs6DKASsFp7zMRGhoqf39/JSUlWddlZWVp69at6tChgwMrAwAA/8uhZyays7O1f/9+6+NDhw4pNTVVPj4+Cg4O1vDhw/X6668rPDxcoaGhGjt2rAIDA9WnTx/HFQ0AAApxaJj4+eef1bVrV+vja9c6DBo0SPPnz9fo0aN14cIFPf300zp37pzuvPNOrVixQh4eHo4qGQAA/IFDw0R0dLQsFkux200mkyZNmqRJkyaVY1UAAMAWTnvNBAAAqBgIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMCQao4uAAAAZ2Y2m5WRkWFXX19fXwUHB5dyRc6HMAEAQDHMZrMiIyOVk5NjV39PT0+lpaVV+kBBmAAAoBgZGRnKycnRK7NeUUjjEJv6Htl/RG++8KYyMjIIEwAAVHUhjUPUpHkTR5fhtLgAEwAAGEKYAAAAhhAmAACAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCHMzVGBpKWl2dwnNzdX7u7uNverKtPmVgT2Tn9cFf6G9vybsKcPgBsjTFQAZ06fkiQNHDiw3I5ZVabNdXZGpj+uzH/D9NNnJBn7N3H27NnSKgeo8ggTFUD2+UxJ0uMvjVO7P91Z4n4/bkjS/Hen2NzPfHCf3hoztEpMm+vsrk1//PnboxQZFlTifmkHj2rgqLcr7d/w3PkLkqTpIx9TdIe2NvVdvv5njZ31qbIvXCiL0oAqiTBRgQQ0aKjwpi1K3N58cJ9d/eB8IsOC1LpZY0eX4XQaBfnbPC5pB4+WUTVA1cUFmAAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEOY6AsAYIjZbFZGRobN/Xx9fe2a1ba8j4ebI0wAAOxmNpsVGRmpnJwcm/t6enoqLS3Npjf48j4eSoYwAQCwW0ZGhnJycvTKrFcU0jikxP2O7D+iN194UxkZGTa9uZf38VAyhAkAgGEhjUPUpHmTSns83BgXYAIAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMceowMWHCBJlMpkLLbbfd5uiyAADA/3D6O2A2a9ZMq1evtj6uVs3pSwYAoEpx+nfmatWqyd/f39FlAACAYjh9mNi3b58CAwPl4eGhDh06KCEh4YaTtOTm5io3N9f6OCsrq0zqsmcK3LS0tDKpBShNTO9ctdn6OsXrGiQnDxN33HGH5s+fr4iICKWnp2vixInq3Lmzdu3apdq1axfZJyEhQRMnTizTuoxMgStJZ8+dLeWKgNLB9M5V15lTZyRJAwcOtKv/2bO8rlVlTh0mYmNjrT+3aNFCd9xxh0JCQvTll1/qySefLLJPfHy8RowYYX2clZWloKCgUq3r2hS4L09JVHBYeIn7/bghSfPfnaIL2RdKtR6gtFx7bn/+9ihFhpX8303awaMaOOptpneuwLKzsiVJT73ylNp1alfiflvXbtXH0z7WhQu8rlVlTh0m/qhOnTpq0qSJ9u/fX2wbd3d3ubu7l0s9wWHhCm/aosTtzQf3lWE1QOmJDAtS62aNHV0GHCAgJMCmqb3N+81lWA0qCqf+augfZWdn68CBAwoICHB0KQAA4D+cOkyMHDlS69at0+HDh7Vp0yY98MADcnV11aOPPuro0gAAwH849cccv/32mx599FH9/vvv8vPz05133qktW7bIz8/P0aUBAID/cOowsWjRIkeXAAAAbsKpP+YAAADOjzABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADKnm6AIAlJ20tLQybV8V2TNGvr6+Cg4OLoNqYCtH/JuwZx+5ublyd3e3uZ+jnmuECaASSj99RpI0cOBAu/qfPXu2NMupFIyMqaenp9LS0ggUDnTmVPn/mzB6THs46rlGmAAqoXPnL0iSpo98TNEd2pa43/L1P2vsrE+VfeFCWZVWYdk7pmkHj2rgqLeVkZFBmHCg7KxsSdJTrzyldp3albjf1rVb9fG0j3XBjn8TRo9pa78j+4/ozRfedMhzjTABVGKNgvzVulnjErdPO3i0DKupHGwdUziXgJAANWnepMTtzfvNDjumrf0ciQswAQCAIYQJAABgCGECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgAAgCGECQAAYAhhAgAAGMLcHACqpJMnT+jAgQM2tD9ZhtUAFRthAkCVcj4rS5L0+ecL9N3ihSXudyyrQJKUlZVZJnUBFRlhAkCVknPxoiTpjq7t1e2e9iXu9833P2r7F5t18T/9AfwXYQJAleRVx1u3Ngy0ob1XGVYDVGxcgAkAAAwhTAAAAEMIEwAAwBDCBAAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQ5uZAqTKbzcrIyLCrb25urtzd3W3u5+vrq+DgYLuOiYrN1mnEJenMmd8NHfPM72dsOubR336TJC1fvlxpaWk2HSssLEwdOnSwqc815v1mm9qnH0236ziARJhAKTKbzYqMjFROTk65HtfT01NpaWkEiirE3mnEpf9OJZ576ZJN/S7lXJ0tdPm//qWNa1eUuN/J7KvHGzt2rE3HkySTpI2bNtkUKNLTr4aCN154w+bjSVLmGaZYh+0IEyg1GRkZysnJ0ctTEhUcFm5T3x83JGn+u1P0+Evj1O5Pd5a4n/ngPr01ZqgyMjIIE1WIvdOIS9LCRWu0fdlO5V3Ot6lfXt7V9s3vaK7e93ex6Xg/Hd+ph7o3UZcubUrcb/eew5r7xWYdPHjQpjBx7tw5SdKoZ2PUpm1Eifslr/5Jc7/YrJzs8v3PACoHwgRKXXBYuMKbtrCpj/ngPklSQIOGNvdF1WXrNOKSVLN2LUPHrOlV06ZjXjteaIifunSNsvFom21s/1/BgT5qGnFridv/vx3/z+5jAVyACQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMAQwgQAADCEMAEAAAypEGEiMTFRDRs2lIeHh+644w79+OOPji4JAAD8h9OHiX/84x8aMWKExo8fr23btqlly5aKiYnRqVOnHF0aAABQBQgT77zzjoYMGaLHH39cTZs21dy5c+Xp6amPP/7Y0aUBAAA5+URfeXl5SklJUXx8vHWdi4uLunfvrs2bi54AJzc3V7m5udbHmZlXp9PN+s+UxaUhOztbkrRsyWL5bNpY4n6H0nZIktYnrZTZfMRp+505eXUK4/fff1/+/v4l7nfixAlJto+LVP61SlK1atV0+fJlm/qUd79rY/rRl8u0op5Piful/npYkrT4X+u1e7+5zPsdO3VGkv3PGVt/vx9+3i1J2vDj/9Pv2bbN/rl999Xn14+pR5Q3799O2+/Q4avP7a+++kppaWkl7rdr1y5J0qrkX7T/cMnP4O7Zc0iS9PPan3Upq+TTsx/89aAkae3StTq061CJ+/1+6ndJ9j9nln66VHXr1S3zOu3t54hjXhvT7OzsUnvPu7Yfi8Vy44YWJ3bs2DGLJMumTZsKrR81apSlffv2RfYZP368RRILCwsLCwtLKS1Hjx694fu1U5+ZsEd8fLxGjBhhfVxQUKAzZ86obt26On/+vIKCgnT06FF5eXk5sMrKKSsri/EtY4xx2WJ8yxbjW7bKYnwtFovOnz+vwMDAG7Zz6jDh6+srV1dXnTx5stD6kydPFntazN3dXe7u7oXW1alTR5JkMpkkSV5eXjyRyxDjW/YY47LF+JYtxrdslfb4ent737SNU1+A6ebmpjZt2igpKcm6rqCgQElJSerQoYMDKwMAANc49ZkJSRoxYoQGDRqktm3bqn379po5c6YuXLigxx9/3NGlAQAAVYAw8cgjj+j06dMaN26cTpw4oaioKK1YsUL169e3eV/u7u4aP378dR+DoHQwvmWPMS5bjG/ZYnzLliPH12Sx3Oz7HgAAAMVz6msmAACA8yNMAAAAQwgTAADAEMIEAAAwpNKFCVunK//qq6902223ycPDQ82bN9fy5cvLqdKKyZbxnT9/vkwmU6HFw8OjHKutWNavX6/evXsrMDBQJpNJS5cuvWmf5ORktW7dWu7u7mrcuLHmz59f5nVWVLaOb3Jy8nXPX5PJZJ0bAoUlJCSoXbt2ql27turVq6c+ffpo7969N+3Ha3DJ2DO+5fkaXKnChK3TlW/atEmPPvqonnzySW3fvl19+vRRnz59rBPloDB7poP38vJSenq6dTlypOSTeFU1Fy5cUMuWLZWYmFii9ocOHdKf//xnde3aVampqRo+fLieeuoprVy5sowrrZhsHd9r9u7dW+g5XK9evTKqsGJbt26dhg4dqi1btmjVqlXKz89Xjx49dOHChWL78BpccvaMr1SOr8GlMyWXc2jfvr1l6NCh1sdXrlyxBAYGWhISEops/5e//MXy5z//udC6O+64w/J///d/ZVpnRWXr+M6bN8/i7e1dTtVVLpIsS5YsuWGb0aNHW5o1a1Zo3SOPPGKJiYkpw8oqh5KM79q1ay2SLGfPni2XmiqbU6dOWSRZ1q1bV2wbXoPtV5LxLc/X4EpzZuLadOXdu3e3rrvZdOWbN28u1F6SYmJiim1fldkzvtLVqXBDQkIUFBSk+++/X7t37y6PcqsEnr/lIyoqSgEBAbrnnnu0ceNGR5dTYWRmZkqSfHyKn1qe57D9SjK+Uvm9BleaMJGRkaErV65cd2fM+vXrF/sZ54kTJ2xqX5XZM74RERH6+OOP9c033+jzzz9XQUGBOnbsqN9++608Sq70inv+ZmVl6eLFiw6qqvIICAjQ3Llz9fXXX+vrr79WUFCQoqOjtW3bNkeX5vQKCgo0fPhwderUSbfffnux7XgNtk9Jx7c8X4Od/nbaqLg6dOhQaEK2jh07KjIyUh988IEmT57swMqAm4uIiFBERIT1cceOHXXgwAHNmDFDn332mQMrc35Dhw7Vrl279MMPPzi6lEqppONbnq/BlebMhD3Tlfv7+9vUviqzZ3z/qHr16mrVqpX2799fFiVWOcU9f728vFSjRg0HVVW5tW/fnufvTcTFxWnZsmVau3atGjRocMO2vAbbzpbx/aOyfA2uNGHCnunKO3ToUKi9JK1atYrpzYtQGtPBX7lyRTt37lRAQEBZlVml8Pwtf6mpqTx/i2GxWBQXF6clS5ZozZo1Cg0NvWkfnsMlZ8/4/lGZvgaXy2We5WTRokUWd3d3y/z58y179uyxPP3005Y6depYTpw4YbFYLJa//vWvlpdfftnafuPGjZZq1apZpk2bZklLS7OMHz/eUr16dcvOnTsd9Ss4NVvHd+LEiZaVK1daDhw4YElJSbH069fP4uHhYdm9e7ejfgWndv78ecv27dst27dvt0iyvPPOO5bt27dbjhw5YrFYLJaXX37Z8te//tXa/uDBgxZPT0/LqFGjLGlpaZbExESLq6urZcWKFY76FZyareM7Y8YMy9KlSy379u2z7Ny50/LCCy9YXFxcLKtXr3bUr+DUnn32WYu3t7clOTnZkp6ebl1ycnKsbXgNtp8941uer8GVKkxYLBbL7NmzLcHBwRY3NzdL+/btLVu2bLFu69Kli2XQoEGF2n/55ZeWJk2aWNzc3CzNmjWzfP/99+VcccViy/gOHz7c2rZ+/fqWe++917Jt2zYHVF0xXPsq4h+Xa2M6aNAgS5cuXa7rExUVZXFzc7OEhYVZ5s2bV+51VxS2ju+UKVMsjRo1snh4eFh8fHws0dHRljVr1jim+AqgqLGVVOg5yWuw/ewZ3/J8DWYKcgAAYEiluWYCAAA4BmECAAAYQpgAAACGECYAAIAhhAkAAGAIYQIAABhCmAAAAIYQJgCUmeTkZJlMJp07d67U920ymbR06dJitx8+fFgmk0mpqak33E90dLSGDx9eqrUBVQ1hAsBNzZ8/X3Xq1HF0GYWkp6crNja2xO3LMtgAVR1TkAOokJhZEnAenJkAqoDo6GjFxcUpLi5O3t7e8vX11dixY3Xtbvq5ubkaOXKkbr31VtWsWVN33HGHkpOTJV39H/3jjz+uzMxMmUwmmUwmTZgwQZL02WefqW3btqpdu7b8/f3Vv39/nTp1yub6LBaL/Pz8tHjxYuu6qKioQrMb/vDDD3J3d1dOTo6k6z/m+PHHH9WqVSt5eHiobdu22r59u3Xb4cOH1bVrV0nSLbfcIpPJpMGDB1u3FxQUaPTo0fLx8ZG/v7/19wNQMoQJoIr45JNPVK1aNf3444+aNWuW3nnnHf3973+XJMXFxWnz5s1atGiRduzYoYcfflg9e/bUvn371LFjR82cOVNeXl5KT09Xenq6Ro4cKUnKz8/X5MmT9csvv2jp0qU6fPhwoTfpkjKZTLrrrrusAebs2bNKS0vTxYsX9euvv0qS1q1bp3bt2snT0/O6/tnZ2erVq5eaNm2qlJQUTZgwwVqjJAUFBenrr7+WJO3du1fp6emaNWtWobGpWbOmtm7dqqlTp2rSpElatWqVzb8HUFXxMQdQRQQFBWnGjBkymUyKiIjQzp07NWPGDMXExGjevHkym80KDAyUJI0cOVIrVqzQvHnz9Oabb8rb21smk+m6jxaeeOIJ689hYWF699131a5dO2VnZ6tWrVo21RcdHa0PPvhAkrR+/Xq1atVK/v7+Sk5O1m233abk5GR16dKlyL4LFy5UQUGBPvroI3l4eKhZs2b67bff9Oyzz0qSXF1d5ePjI0mqV6/eddd/tGjRQuPHj5ckhYeH67333lNSUpLuuecem34HoKrizARQRfzpT3+SyWSyPu7QoYP27dunnTt36sqVK2rSpIlq1aplXdatW6cDBw7ccJ8pKSnq3bu3goODVbt2beubvdlstrm+Ll26aM+ePTp9+rTWrVun6OhoRUdHKzk5Wfn5+dq0aZOio6OL7JuWlqYWLVrIw8Oj0O9XUi1atCj0OCAgwK6Pa4CqijMTQBWXnZ0tV1dXpaSkyNXVtdC2G51duHDhgmJiYhQTE6MFCxbIz89PZrNZMTExysvLs7mO5s2by8fHR+vWrdO6dev0xhtvyN/fX1OmTNFPP/2k/Px8dezY0eb9lkT16tULPTaZTCooKCiTYwGVEWECqCK2bt1a6PGWLVsUHh6uVq1a6cqVKzp16pQ6d+5cZF83NzdduXKl0Lpff/1Vv//+u9566y0FBQVJkn7++We76zOZTOrcubO++eYb7d69W3feeac8PT2Vm5urDz74QG3btlXNmjWL7BsZGanPPvtMly5dsp6d2LJly3W/g6Trfg8AxvExB1BFmM1mjRgxQnv37tUXX3yh2bNn64UXXlCTJk00YMAAPfbYY/rnP/+pQ4cO6ccff1RCQoK+//57SVLDhg2VnZ2tpKQkZWRkKCcnR8HBwXJzc9Ps2bN18OBBffvtt5o8ebKhGqOjo/XFF18oKipKtWrVkouLi+666y4tWLCg2OslJKl///4ymUwaMmSI9uzZo+XLl2vatGmF2oSEhMhkMmnZsmU6ffq0srOzDdUK4L8IE0AV8dhjj+nixYtq3769hg4dqhdeeEFPP/20JGnevHl67LHH9NJLLykiIkJ9+vTRTz/9pODgYElSx44d9cwzz+iRRx6Rn5+fpk6dKj8/P82fP19fffWVmjZtqrfeeuu6N3BbdenSRVeuXCl0bUR0dPR16/6oVq1a+u6777Rz5061atVKr776qqZMmVKoza233qqJEyfq5ZdfVv369RUXF2eoVgD/ZbJc+6I5gEorOjpaUVFRmjlzpqNLAVAJcWYCAAAYQpgAUC5iY2MLffX0f5c333zT0eUBMICPOQCUi2PHjunixYtFbvPx8bHeVApAxUOYAAAAhvAxBwAAMIQwAQAADCFMAAAAQwgTAADAEMIEAAAwhDABAAAMIUwAAABDCBMAAMCQ/w9rAMX7t/zYWgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "hist_graphing('petal_width', 30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhkAAAIPCAYAAADelcwNAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACXn0lEQVR4nOzdd3zT1frA8U+aNqPN6G7pBKQie28BES8qDhSvA1EU0Z/zuhdexateRa963aKggl5B3HtvlKWylwgIlNG90jR7/P6IFEKS0pWkLc/79epLe87JydPQ5vvkfM9QeL1eL0IIIYQQrSwm2gEIIYQQomOSJEMIIYQQYSFJhhBCCCHCQpIMIYQQQoSFJBlCCCGECAtJMoQQQggRFpJkCCGEECIsJMkQQgghRFhIkiGEEEKIsJAkQwghhBBhERvtAA54+OGHmTlzJjfccANPPvlk0DYLFixg+vTpfmVqtRqbzdbo5/F4POzfvx+9Xo9CoWhJyEIIIcRRxev1UltbS1ZWFjExRx6naBNJxq+//sqLL75I3759j9jWYDCwdevW+u+bmijs37+f3NzcJscohBBCCJ89e/aQk5NzxHZRTzLMZjNTp05l3rx5/Pvf/z5ie4VCQWZmZrOfT6/XA74XyGAwNLsfIYQQ4mhjMpnIzc2tv5YeSdSTjGuvvZbTTjuNk046qVFJhtlsJj8/H4/Hw8CBA3nooYfo1atXyPZ2ux273V7/fW1tLeAbEZEkQwghhGi6xt5FiOrEz8WLF7N69Wpmz57dqPbdu3fnlVde4cMPP+T111/H4/EwcuRI9u7dG/Ixs2fPxmg01n/JrRIhhBAiMhRer9cbjSfes2cPgwcP5uuvv66fi3HCCSfQv3//kBM/D+d0OunRowdTpkzhgQceCNrm8JGMA0M9NTU1MpIhhBBCNIHJZMJoNDb6Ghq12yWrVq2itLSUgQMH1pe53W6WLFnCs88+i91uR6lUNthHXFwcAwYMYPv27SHbqNVq1Gp1q8UthBBCiMaJWpIxfvx4NmzY4Fc2ffp0jjvuOO64444jJhjgS0o2bNjAxIkTwxWmEEKIRnC73TidzmiHIVpBXFxco67BjRG1JEOv19O7d2+/soSEBFJSUurLp02bRnZ2dv2cjfvvv5/hw4fTrVs3qqurefTRR9m9ezeXX355xOMXQgjhYzab2bt3L1G6+y5amUKhICcnB51O1+K+or66pCGFhYV+m31UVVVxxRVXUFxcTFJSEoMGDWLZsmX07NkzilEKIcTRy+12s3fvXuLj40lLS5NNDts5r9dLWVkZe/fupaCgoMUjGlGb+BktTZ20IoQQIjSbzcbOnTvp3LkzWq022uGIVmC1Wtm1axddunRBo9H41TX1GipnlwghhGgxGcHoOFrz31KSDCGEEEKEhSQZQgghRAtdeumlnHXWWdEOo81p0xM/hRBCiPbgqaeektU1QUiSIYQQQrSQ0WiMdghtktwuEUII0SG888479OnTB61WS0pKCieddBJ1dXX1tzLuu+8+0tLSMBgMXHXVVTgcjvrHejweZs+eTZcuXdBqtfTr14933nnHr/9NmzZx+umnYzAY0Ov1jB49mh07dgCBt0uO1F9VVRVTp04lLS0NrVZLQUEB8+fPD+8LFAUykiFEB+J2eyiptVNtcRCnjCE5QUWKTrbVFx1fUVERU6ZM4T//+Q9nn302tbW1/PTTT/W3ML799ls0Gg0//PADu3btYvr06aSkpPDggw8CvsM0X3/9dV544QUKCgpYsmQJF110EWlpaYwdO5Z9+/YxZswYTjjhBL777jsMBgNLly7F5XIFjedI/d1zzz1s3ryZzz//nNTUVLZv347Vao3Y6xUpkmQI0UGYrE6+2lzCvz/dTLXFt71zz04Gnji/P8dm6GSJoejQioqKcLlcTJ48mfz8fAD69OlTX69SqXjllVeIj4+nV69e3H///dx222088MADOJ1OHnroIb755htGjBgBQNeuXfn555958cUXGTt2LM899xxGo5HFixcTFxcHwLHHHhs0FrvdfsT+CgsLGTBgAIMHDwagc+fO4XppokqSDCE6iDWFVdz69jq/ss1FJs6fu5xP/nE8OUnxUYpMiPDr168f48ePp0+fPpx88slMmDCBv//97yQlJdXXx8cf/BsYMWIEZrOZPXv2YDabsVgs/O1vf/Pr0+FwMGDAAADWrl3L6NGj6xOMhmzfvv2I/V199dWcc845rF69mgkTJnDWWWcxcuTIFr0GbZEkGUJ0ABVmOw9/8XvQumqLk+U7Kjh3sCQZouNSKpV8/fXXLFu2jK+++opnnnmGf/7zn6xcufKIjzWbzQB8+umnZGdn+9UdOMW7KbuZNqa/U089ld27d/PZZ5/x9ddfM378eK699loee+yxRj9PeyBJhhAdgMPlYWtxbcj6X3ZWcu7g3AhGJETkKRQKRo0axahRo5g1axb5+fm8//77AKxbtw6r1VqfLKxYsQKdTkdubi7Jycmo1WoKCwsZO3Zs0L779u3Lq6++itPpPOJoRs+ePY/YH0BaWhqXXHIJl1xyCaNHj+a2226TJEMI0fYolQqyErXsrQo+cax7pj7CEQkRWStXruTbb79lwoQJpKens3LlSsrKyujRowfr16/H4XAwY8YM7r77bnbt2sW9997LddddR0xMDHq9nltvvZWbbroJj8fD8ccfT01NDUuXLsVgMHDJJZdw3XXX8cwzz3DBBRcwc+ZMjEYjK1asYOjQoXTv3t0vlsb0N2vWLAYNGkSvXr2w2+188skn9OjRI0qvXvhIkiFEB5Cu13DD+AJue2d9QJ1KGcNJPTOiEJUQkWMwGFiyZAlPPvkkJpOJ/Px8Hn/8cU499VTefPNNxo8fT0FBAWPGjMFutzNlyhT+9a9/1T/+gQceIC0tjdmzZ/Pnn3+SmJjIwIEDueuuuwBISUnhu+++47bbbmPs2LEolUr69+/PqFGjgsZzpP5UKhUzZ85k165daLVaRo8ezeLFi8P+OkWanMIqRAdRYbbz/A/bmb90F56//qoNmlhevHgQgzonoWrhkc1CBHPgFNZgJ3a2FZdeeinV1dV88MEH0Q6lXWjo37Sp11AZyRCig0jRqbnppGO5eHhndlfUEa+OJTtRS7peTaxS9t0TQkSeJBlCdCA6TRw6TRydUxOiHYoQQkiSIYQQomNbsGBBtEM4askYqhBCCCHCQpIMIYQQQoSFJBlCCCGECAtJMoQQQggRFpJkCCGEECIsJMkQQgghRFhIkiGEEEKIsJAkQwghhGiCXbt2oVAoWLt2bbRDafNkMy4hhBBR5/Z4+WVnJaW1NtL1GoZ2SUYZo4h2WKKFZCRDCCFEVH2xsYjjH/mOKfNWcMPitUyZt4LjH/mOLzYWhfV533nnHfr06YNWqyUlJYWTTjqJuro6AF566SV69OiBRqPhuOOO4/nnn69/XJcuXQAYMGAACoWCE044AQCPx8P9999PTk4OarWa/v3788UXX9Q/zuFwcN1119GpUyc0Gg35+fnMnj27vv6///0vffr0ISEhgdzcXK655hrMZnNYX4NwkyRDCCFE1HyxsYirX19NUY3Nr7y4xsbVr68OW6JRVFTElClTuOyyy9iyZQs//PADkydPxuv1snDhQmbNmsWDDz7Ili1beOihh7jnnnt49dVXAfjll18A+OabbygqKuK9994D4KmnnuLxxx/nscceY/369Zx88smceeaZbNu2DYCnn36ajz76iLfeeoutW7eycOFCOnfuXB9TTEwMTz/9NJs2beLVV1/lu+++4/bbbw/Lzx8pctS7EEKIZmvJUe9uj5fjH/kuIME4QAFkGjX8fMeJrX7rZPXq1QwaNIhdu3aRn5/vV9etWzceeOABpkyZUl/273//m88++4xly5axa9cuunTpwpo1a+jfv399m+zsbK699lruuuuu+rKhQ4cyZMgQnnvuOa6//no2bdrEN998g0Jx5J/nnXfe4aqrrqK8vLzlP3ATtOZR7zKSIYQQIip+2VkZMsEA8AJFNTZ+2VnZ6s/dr18/xo8fT58+fTj33HOZN28eVVVV1NXVsWPHDmbMmIFOp6v/+ve//82OHTtC9mcymdi/fz+jRo3yKx81ahRbtmwB4NJLL2Xt2rV0796d66+/nq+++sqv7TfffMP48ePJzs5Gr9dz8cUXU1FRgcViafWfP1IkyRBCCBEVpbWhE4zmtGsKpVLJ119/zeeff07Pnj155pln6N69Oxs3bgRg3rx5rF27tv5r48aNrFixokXPOXDgQHbu3MkDDzyA1WrlvPPO4+9//zvgW7Fy+umn07dvX959911WrVrFc889B/jmcrRXsrpECCFEVKTrG3d7pbHtmkqhUDBq1ChGjRrFrFmzyM/PZ+nSpWRlZfHnn38yderUoI9TqVQAuN3u+jKDwUBWVhZLly5l7Nix9eVLly5l6NChfu3OP/98zj//fP7+979zyimnUFlZyapVq/B4PDz++OPExPg+/7/11lvh+LEjSpIMIYQQUTG0SzKdjBqKa2wEmxx4YE7G0C7Jrf7cK1eu5Ntvv2XChAmkp6ezcuVKysrK6NGjB/fddx/XX389RqORU045Bbvdzm+//UZVVRU333wz6enpaLVavvjiC3JyctBoNBiNRm677TbuvfdejjnmGPr378/8+fNZu3YtCxcuBHyrRzp16sSAAQOIiYnh7bffJjMzk8TERLp164bT6eSZZ57hjDPOYOnSpbzwwgut/nNHmiQZQgghokIZo+DeM3py9eurUYBfonFgWuS9Z/QMy34ZBoOBJUuW8OSTT2IymcjPz+fxxx/n1FNPBSA+Pp5HH32U2267jYSEBPr06cONN94IQGxsLE8//TT3338/s2bNYvTo0fzwww9cf/311NTUcMstt1BaWkrPnj356KOPKCgoAECv1/Of//yHbdu2oVQqGTJkCJ999hkxMTH069eP//73vzzyyCPMnDmTMWPGMHv2bKZNm9bqP3skyeoSIYQQzdaS1SUHfLGxiPs+3uw3CbSTUcO9Z/TklN6dWitU0UitubpERjKEEEJE1Sm9O/G3npmy42cHJEmGEEKIqFPGKBhxTEq0wxCtTJawCiGEECIsJMkQQgghRFhIkiGEEEKIsJAkQwghhBBhIRM/hYiwGquDCrMDm9ONQRNHml6NOk4Z7bCEEKLVSZIhRATtrbJwxzvrWbqjAgBNXAxXjjmGaSPySdGpoxydEEK0LrldIkSElJpsXDr/1/oEA8Dm9PDUt9t4+7c9uNyeKEYnhBCtT5IMISJkT5WV7aXmoHXP/7CD0lp7hCMSQkTKrl27UCgUrF27tk32Fy5yu0SICNleWhuyzmRzYXG4IhiNECKScnNzKSoqIjU1NdqhRJQkGUJESE5SfMg6dWwM6liZ/CmOYh437F4G5hLQZUD+SIhpP38TTqeTuLi4kPVKpZLMzMwIRnRkDoej/tj6cJHbJUJESJfUBNJCTO48b3AuaXqZ+CmOUps/gid7w6unw7szfP99srevPAzmzp1LVlYWHo//PKhJkyZx2WWXAfDhhx8ycOBANBoNXbt25b777sPlOjjaqFAomDNnDmeeeSYJCQk8+OCDVFVVMXXqVNLS0tBqtRQUFDB//nwg+O2NTZs2cfrpp2MwGNDr9YwePZodO3YA4PF4uP/++8nJyUGtVtO/f3+++OKLBn+uH3/8kaFDh6JWq+nUqRN33nmnX8wnnHAC1113HTfeeCOpqamcfPLJLXodG0OSDCEiJCtRy+uXD6OT0f9Uw5N6pHPdid3QyDJWcTTa/BG8NQ1M+/3LTUW+8jAkGueeey4VFRV8//339WWVlZV88cUXTJ06lZ9++olp06Zxww03sHnzZl588UUWLFjAgw8+6NfPv/71L84++2w2bNjAZZddxj333MPmzZv5/PPP2bJlC3PmzAl5e2Tfvn2MGTMGtVrNd999x6pVq7jsssvqk4KnnnqKxx9/nMcee4z169dz8sknc+aZZ7Jt27aQ/U2cOJEhQ4awbt065syZw8svv8y///1vv3avvvoqKpWKpUuX8sILL7TkZWwc71GmpqbGC3hramqiHYo4ShVVW7xrC6u8320p8W4rMXmr6uzRDkmIZrNard7Nmzd7rVZr0x/sdnm9jx/n9d5rCPFl9Hof7+Fr18omTZrkveyyy+q/f/HFF71ZWVlet9vtHT9+vPehhx7ya/+///3P26lTp/rvAe+NN97o1+aMM87wTp8+Pejz7dy50wt416xZ4/V6vd6ZM2d6u3Tp4nU4HEHbZ2VleR988EG/siFDhnivueaaoP3ddddd3u7du3s9Hk99++eee86r0+m8brfb6/V6vWPHjvUOGDAg1EtSr6F/06ZeQ2UkQ4gIyzRq6ZebyLjj0umWricxPrz3RIVos3YvCxzB8OMF0z5fu1Y2depU3n33Xex236quhQsXcsEFFxATE8O6deu4//770el09V9XXHEFRUVFWCyW+j4GDx7s1+fVV1/N4sWL6d+/P7fffjvLloWOe+3atYwePTroPA6TycT+/fsZNWqUX/moUaPYsmVL0P62bNnCiBEjUCgUfu3NZjN79+6tLxs0aFADr0rrkyRDCCFEdJhLWrddE5xxxhl4vV4+/fRT9uzZw08//cTUqVN9T2c2c99997F27dr6rw0bNrBt2zY0moO3OxMSEvz6PPXUU9m9ezc33XQT+/fvZ/z48dx6661Bn1+r1bb6z9QYh8ccbpJkCCGEiA5dRuu2awKNRsPkyZNZuHAhb7zxBt27d2fgwIEADBw4kK1bt9KtW7eAr5iYhi+baWlpXHLJJbz++us8+eSTzJ07N2i7vn378tNPP+F0OgPqDAYDWVlZLF261K986dKl9OzZM2h/PXr0YPny5Xi9Xr/2er2enJycBmMOJ1nCKoQQIjryR4IhyzfJE2+QBgpfff7IsDz91KlTOf3009m0aRMXXXRRffmsWbM4/fTTycvL4+9//3v9LZSNGzcGTKQ81KxZsxg0aBC9evXCbrfzySef0KNHj6Btr7vuOp555hkuuOACZs6cidFoZMWKFQwdOpTu3btz2223ce+993LMMcfQv39/5s+fz9q1a1m4cGHQ/q655hqefPJJ/vGPf3DdddexdetW7r33Xm6++eYjJkbhJCMZQgghoiNGCac88tc3isMq//r+lIfDtl/G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",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(6,6))\n",
    "sns.scatterplot(data = iris, x = 'sepal_length', y = 'sepal_width', hue = 'species')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "def scatter_line(df, x, y, figsize):\n",
    "    p1 = df[x]\n",
    "    p2 = df[y]\n",
    "    \n",
    "    plt.figure(figsize=(figsize,figsize))\n",
    "    plt.scatter(data = df, x = x, y = y)\n",
    "    \n",
    "    z = np.polyfit(p1, p2, 1)\n",
    "    p = np.poly1d(z)\n",
    "    plt.plot(p1, p(p1))\n",
    "    \n",
    "    # plt.title(df)\n",
    "    # plt.xlabel(x)\n",
    "    # plt.ylabel(y)\n",
    "    \n",
    "    plt.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_setosa = iris[iris.species == 'setosa']\n",
    "iris_versicolor = iris[iris.species == 'versicolor']\n",
    "iris_virginica = iris[iris.species == 'virginica']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_species = [iris_setosa, iris_versicolor, iris_virginica]\n",
    "species = ['setosa', 'versicolor', 'virginica']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "setosa correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "versicolor correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "virginica correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(len(iris_species)):\n",
    "    print(f\"{species[i]} correlation visual\")\n",
    "    scatter_line(iris_species[i], 'sepal_length', 'sepal_width', 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "setosa correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "versicolor correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "virginica correlation visual\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAR4AAAESCAYAAAArC7qtAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAhMUlEQVR4nO3dfVxUdd438M8gy0AJk1jIY0KWD0SZ5XqHKD5UKhp21XW7rqZgupYKletlW7SVsLs54rW3r311VZZs4frAeq9ePqSXUpYKmpia0kakaYKiAmtlMzwkKnPuP7iHHGBmzgxnfnPOzOf9es0fzPzOOb9zYL6ch9/5HJ0kSRKIiAQK8HYHiMj/sPAQkXAsPEQkHAsPEQnHwkNEwrHwEJFwLDxEJFygtzsgh8ViwcWLFxEaGgqdTuft7hDRDSRJQkNDA6KjoxEQIG9fRhOF5+LFi4iLi/N2N4jIgZqaGsTGxspqq4nCExoaCqBtxcLCwrzcGyK6kdlsRlxcXPv3VA5NFB7r4VVYWBgLD5FKuXIahCeXiUg4Fh4iEo6Fh4iE08Q5HvIdrRYJh6t+wL8ariAiNBjDEsLRI8AzQyScLUtkX8gWCw8JU1xRi7ztlag1XWl/L8oQjCXpiZiQFCV0WSL7Qp3ptBAEZjabYTAYYDKZeFVLo4orajF/3TF0/GOz7l+snHG/Yl94Z8t6OjUBq0qrhPTFH7jz/eQ5HvK4VouEvO2Vnb7oANrfy9teiVZL9/8HOluWBKBgf+ei44m+kH0sPORxh6t+sDmk6UgCUGu6gsNVP3h8WQDgqKYo2Reyj4WHPO5fDY4LgavtPD0PJedDXWPhIY+LCA1WtJ2n56HkfKhrLDzkccMSwhFlCIa9C9U6tF1RGpYQ7vFlAUCADkL6Qvax8JDH9QjQYUl6IoDOX3jrz0vSExUZQ+NsWToAc0cmCOkL2cfCQ0JMSIrCyhn3I9JgewgTaQhW/PK1s2XlTEwU1hfqGsfxkFAcuex73Pl+cuQyCdUjQIfkfr1VsSyRfSFbPNQiIuFYeIhIOBYeIhKOhYeIhGPhISLhWHiISDgWHiISjuN4VExrA9yuXrdgbVk1zv7QjL7hN2FmcjyCAvm/jTpzqfAYjUZs3rwZJ06cQEhICIYPH478/HwMGDBA1vQbNmzAtGnT8Nhjj2Hr1q3u9NdvaC2a07izEgX7q2yybl7f+TXmjkxAzsRE73WMVMmlf0clJSXIysrCoUOHsHv3bly7dg3jxo1DU1OT02mrq6uxePFijBw50u3O+gtrdGfHQKs60xXMX3cMxRW1XupZ14w7K/FuaVWngC2LBLxbWgXjzkrvdIxUq1v3al26dAkREREoKSlBamqq3Xatra1ITU3F7NmzsX//fvz4448u7fH4071arRYJI/L32E3R06HtZsYDL45VxWHX1esWDHx1l8NUvwAdcOKPaTzs8lHCM5dNJhMAIDzccXbJH/7wB0RERGDOnDmy5tvS0gKz2Wzz8hciY0KVsLas2mHRAdr2fNaWVQvpD2mD24XHYrFg4cKFSElJQVJSkt12Bw4cwHvvvYeCggLZ8zYajTAYDO2vuLg4d7upOSJjQpVw9odmRduRf3C78GRlZaGiogIbNmyw26ahoQEzZ85EQUEBbr31VtnzzsnJgclkan/V1NS4203NERkTqoS+4Tcp2o78g1uX07Ozs7Fjxw6UlpYiNjbWbrtvv/0W1dXVSE9Pb3/PYrG0LTgwECdPnkS/fv06TafX66HX693pmuZZozvrTFe6fASL9RyPWqI5ZybH4/WdXzs9xzMzOV5Yn0j9XNrjkSQJ2dnZ2LJlC/bs2YOEhASH7QcOHIgvv/wS5eXl7a/JkydjzJgxKC8v96tDKLlExoQqISgwoD1K1J65IxN4YplsuLTHk5WVhaKiImzbtg2hoaGoq6sDABgMBoSEhAAAMjIyEBMTA6PRiODg4E7nf2655RYAcHheyN9Zozs7juOJVOk4Hus4nY7jeAJ04Dge6pJLhWflypUAgNGjR9u8X1hYiFmzZgEAzp07h4AA/nfrrglJUXgkMVIzI5dzJibiP8YN5MhlkoWZy0TULXx2OhFpAgsPEQnHwkNEwrHwEJFwLDxEJBwLDxEJx8JDRMIx+lTF5ESfimwjap1EzUdr0bIieXrbsPColJzoU5FtRK2TqPloLVpWJBHbhiOXVcgafdrxF2P9f7Nyxv0AIKyNEn9sctZJznKUmI9SffFF7mwbd76fLDwqIyf6tE+YHoAOdWbH8aiSJKHO3NLt+XQ3ZlWpOFcl5qO1aFmR3N02vGXCB8iJPq0zt9gtFtY2taYrdouOq/PpbsyqUnGuSsxHa9GyIoncNiw8KqOWSNMbdbdPSsW5KjEfrUXLiiRy27DwqIxaIk1v1N0+KRXnqsR8tBYtK5LIbcPCozLW6FN7Zxd0ACLD9IgMc9wmyhCMyDC9IvPpbsyqnHWSsxwl5qNUX3yRyG3DwqMycqJPcyffjdzJzuNRcyffrch8unuSVak4VyXmo7VoWZFEbhsWHhWyRp9GGmx3aSMNwe2XM0W2EbVOouYjap21SNS24eV0FePIZc/OhyOX7XNl23AcDxEJx3E8RKQJLDxEJBwLDxEJx7vTiahdU8t1/Ol/vsbfD58D0PZAxt9PUv6BjCw8RH7uWqsFb3xyCv+153Snz5qvtnpkmSw8RH7IYpGwpqwaudsr7bbJTU9ERnK8R5bPwkPkR7Z/cRHP/v243c+fHXsnnh17l8cfPc3CQ4pR0+BAf9XVtjt05ns8veYomuwcNk0bdjtemTQIN+vFlQOXlmQ0GrF582acOHECISEhGD58OPLz8zFgwAC70xQUFGDNmjWoqKgAADzwwANYunQphg0b1r2ek6qoKdbUX3W17ewZl9gHxifuQe+eegE968yl/amSkhJkZWXh0KFD2L17N65du4Zx48ahqanJ7jT79u3DtGnTsHfvXpSVlSEuLg7jxo3DhQsXut15UgdrXGbHP/g60xXMX3cMxRW1Qufjj4orajGvi213owf69sL+341B9bJJWJUx1GtFB+jmLROXLl1CREQESkpKkJqaKmua1tZW9OrVC2+++SYyMjJkTcNbJtRLTbGm/uhEnRkT/rLfabsoD247d76f3TqoM5lMAIDwcPn5HM3Nzbh27ZrDaVpaWtDS8nNsp9lsdr+T5FGuxGUm9+vt8fn4g3rzFfyvpZ+4NI3atp3bhcdisWDhwoVISUlBUlKS7OlefPFFREdH4+GHH7bbxmg0Ii8vz92ukUBqijX1Zabmaxj8h4+6NQ81bTu3C09WVhYqKipw4MAB2dMsW7YMGzZswL59+xAcbD8+MScnB4sWLWr/2Ww2Iy4uzt2ukgepKdbU11xrteCu3+9y2GZgZChefTQRT/71M6fzU9O2c6vwZGdnY8eOHSgtLUVsbKysaf785z9j2bJl+Pjjj3Hvvfc6bKvX66HXe+/EF8lnjcusM13p9Cwm4OdzM3JjTbs7H62TJAkJOTudtjvxxwkI/kUPAG3nx7S27VwqPJIk4dlnn8WWLVuwb98+JCQkyJpu+fLleP311/Hhhx9i6NChbnWU1Mkalzl/3THoAJs/fHdiTbs7H62a8JdSnKhrcNjmwItjENvrpk7va3HbuXRVa8GCBSgqKsK2bdtsxu4YDAaEhIQAADIyMhATEwOj0QgAyM/Px2uvvYaioiKkpKS0T9OzZ0/07NlT1nJ5VUv9OI7HdS9u+if+79Eah23+e/5wPNC3l6z5eWvbeTyBUKfrumIWFhZi1qxZAIDRo0cjPj4eq1evBgDEx8fj7NmznaZZsmQJcnNzZS2XhUcbOHLZucJPq5Dn4P4oAPjL1Pvwb0Ni3Jq/N7Ydo0+JVGjT5+exeOMXDttkjemHF8YPFNQjZQkfx0NEXTt27jKeePugwzajB9yG1U/5561DLDxECqkzXcGDRucD+6qXTRLQG3Vj4SHqhuar15H42odO251ZOhEBPnKeSgksPEQukjvW5svccQgN/oWAHmkPCw+RTPEv/Y/TNvsWj0b8rTcL6I22sfAQOSCn2PzXtCFIHxwtoDe+g4WHqIMp7xzEkerLDtvMGZGAVx9V/ukL/oKFhwA4H3imtme0Kz1QbsVHJ/FGF09ZuNGgqDDsen6k28voLl8aWMnCQ06H2ssZiq9UGyX6K9fqT6scPmXBSg2Xv33tVhKOXPZz1rjRjn8E1v+jT6cmYFVpld3PV864HwAczkNuGzlfIGf9dTaf4+cu43EnA/sAdRQbq+6us6fxlglyiZy4UZ0OsNj5C9EB6BOmB6BDndlxZKkkSagztzhs46l41MtNVzHkj7vtztdKjWNttBAJy1smyCVy4kYd/VuSALvF5MY2zp564Il41GEJ4ej3svOxNsdefQThNwc5bectvhoJy8Ljx9QUhQkoF486reCQw883LxiO+2+XFzXhbb4aCcvC48fUFIUJKBeP2pUl6Yl4KkVecJ2a+GokLAuPH5MTNyr3HE+92XHspiRJqDe3eDQetaO48BDs/91YGS3Vy1cjYT37gGRSNWtkJvDzFRIr689zRya0FSA7n+dOvhu5kx3PY0l6InIn3+20jbOTo/1e3olaGUWnetkkVC+bpPmiA8j7Hakt1lQOXtUiVY/jearwMPaevOR0HbQ8pkUONY/j4eV0cpuaRi7v+OdFZBcdd9rnv8990CdG8cql1pHLLDykWee+b0bqf+512q7KONFu9jd5B8fxkKbIeWAdABx95WHc2pPPWfMlLDwknJyoib/PfVBTA+LINSw8JIScYpM95k4sHj/AaTvSPhYe8hg5xSbKEIyynIcE9IbUhIWHFCWn2ADquvubxGPhoW577M0D+OK8yWk7FhuyYuEht2gpRIvUh4XHQ9Q0IE8pX543If3NA07bcaxNZ6J+T2odZNiRS4XHaDRi8+bNOHHiBEJCQjB8+HDk5+djwADHVyI2btyIV199FdXV1bjrrruQn5+PiRMndqvjaqbmWxBc9dPVVgx6rdhpuy+WjIMhhM+Q6oqo2x3UfFtFRy6NXJ4wYQJ+/etf45e//CWuX7+Ol19+GRUVFaisrMTNN3f9LKGDBw8iNTUVRqMRjz76KIqKipCfn49jx44hKSlJ1nK1NHJZa1Gi9sg5SbxpXjKGxmvrrmjRRMWWejMeVfgtE5cuXUJERARKSkqQmpraZZupU6eiqakJO3bsaH/vwQcfxH333Yd33nlH1nK0UniUiBKVExMqN27U1ThMOcWGj3WRT1RsqbfjUYXfMmEytV3JCA+3/1+vrKwMixYtsnlv/Pjx2Lp1q91pWlpa0NLy8xfPbDZ3p5vCKBElKicmVG7cqJw4TDnFJjBAh9NLfffQ2FNExZZqMR7V7cJjsViwcOFCpKSkODxkqqurQ58+fWze69OnD+rq6uxOYzQakZeX527XvEZt8ZP2+sOxNmKIii3VYjyq24UnKysLFRUVOHDA+VUOV+Xk5NjsJZnNZsTFxSm+HKWpLX7yxv6w2IgnKrZUi/GobhWe7Oxs7NixA6WlpYiNjXXYNjIyEvX19Tbv1dfXIzIy0u40er0eer327kZWIkpUTkyo3LjRnV/WOg0+B1hsPEVUbKkW41Fdij6VJAnZ2dnYsmUL9uzZg4QE5+HZycnJ+OSTT2ze2717N5KTk13rqQYoESUqJybUUdwo8PMx/dpDZ+321RoPyqLjOaJiS7UYj+rSVa0FCxagqKgI27Ztsxm7YzAYEBISAgDIyMhATEwMjEYjgLbL6aNGjcKyZcswadIkbNiwAUuXLvXZy+mAd8fxOFKRNx499RwzKpqvj+Px+OV0e6NRCwsLMWvWLADA6NGjER8fj9WrV7d/vnHjRrzyyivtAwiXL1/u0gBCrRUeQMzIZTnnbf57fjIe6KueXWx/5csjlxl96gfkFJt5o/rhpbSBAnpDxOhTnyWn2AT1CMA3r6cJ6A1R97HwqBQvf5MvY+FRkZnvfYb9p75z2o7FhrSOhcfLmGtD/oiFxwtqfmjGyOXOnyHFYkO+ioVHELm5NqdeT8MvevCR9uTbWHg8SJIkJOTsdNru05fGIuaWEAE9IlIHFh4PGPt/9uHMpSaHbbZnj0Bjy3X8q+EKzn3fjMiw4E4Dva5et2BtWTXO/tCMvuE3YWZyPIICXd8b0kocpmjcLt7DwqOQhRuOY2v5RYdt/jL1PvzbkBgUV9Ti6bVHHQ5tN+6sRMH+KpsbSl/f+TXmjkxAzkT5QVxaisMUidvFuzhyuRveLfkWxl0nHLaZNTy+/aZPQF5E5fFzl/FuaZXdeT6TKq/4eDMOU824XZTFWyYEqDdfwYL1x/D52ct22zz30F1Y9Ej/Tu/Ljai0F29gFaADTvwxzeFhl7fjMNWK20V5vGXCQ0w/XUPuB19hy/ELdts8PKgP/po51OF85EZUOmORgLVl1Zgz8o5uL0tNcZgicLuoAwuPHVeuteLPH57EXw/YP+Tp36cnPvrtKNnzVDJ68uwPzYosS01xmCJwu6gDC88NWi0SVpWeQX6x/fM2+f9+D341NM6tB9YpGT3ZN/wmRZalpjhMEbhd1MHvC48kSdj4+Xn8btM/7bZ5YfwAzBvVr9vH/HIjKuWc45mZHK/IstQUhykCt4s6+G3h+eTresxdc9Ru/vHslAT8bsIABP+ih2LLtEZUzl93DDrA5g//xohKZ1e15o5McDqeR+6y/O0EKreLOvjVVa3Pz/6AZ9Z+ju8ar3b5+eNDYpA7+W6PP4pXzhiSrsbxBOjAcTwK4XZRDi+nd+FUfQOyi47jZH1Dl5+PvOtW/Of/HoxIg9hjejmjZjly2bO4XZTBwvP/1Zp+wn/84wsc/Pb7Lj9PjArDG9OG4M6Inkp3lcjv+P04nvOXmzEiv+u4iT5heqyc8QDuv72X4F4RUUc+VXg6PkcqMECHVRkPYOzAPnamICJv8KnC85sRd6C1VcLAqDD8+/0xbo21ISLP86nCc1uoHq88Kv+KDxF5B6PuiEg4Fh4iEo6Fh4iE86lzPKLIGdinxOA/NQ1wk7M+SvVXbfNRy3J8iU8OIPQkObcyKHG7g5qG9MtZH6X6q7b5qGU5aubO99PlQ63S0lKkp6cjOjoaOp0OW7dudTrN+vXrMXjwYNx0002IiorC7Nmz8f33XY8qVjPjzkq8W1rV6cZSiwS8W1oF485KWW2csUZzdgysqjNdwfx1x1BcUdvdVZFNzvoo1V+1zUcty/FFLheepqYmDB48GG+99Zas9p9++ikyMjIwZ84cfPXVV9i4cSMOHz6MuXPnutxZb7p63YKC/fbvGAeAgv1VWOXgrnJrm6vXLXY/b7VIyNte2WVkg/W9vO2VaLV3W72C5K7zkm0V3e6vUustavup6fekRS4XnrS0NPzpT3/C448/Lqt9WVkZ4uPj8dxzzyEhIQEjRozAM888g8OHD9udpqWlBWaz2eblbWvLqu1GaFhZJDjM0bG2WVtWbfdzV6I5PU3uOtc3dH23PyC/v0qtt6jtp6bfkxZ5/KpWcnIyampqsHPnTkiShPr6emzatAkTJ060O43RaITBYGh/xcXFebqbTjmLGlVqXmqK5lRynZ31V6n1FrX91PR70iKPF56UlBSsX78eU6dORVBQECIjI2EwGBwequXk5MBkMrW/ampqPN1Np5xFjSo1LzVFcyq5zs76q9R6i9p+avo9aZHHC09lZSWef/55vPbaa/j8889RXFyM6upqzJs3z+40er0eYWFhNi9vm5kcD2dXSAN0P6fYOWrjKLbUGs1pbz46tF01ERHNKXed+4QGdbu/Sq23qO2npt+TFnm88BiNRqSkpOCFF17Avffei/Hjx+Ptt9/G+++/j9pa7Zz1DwoMwNyRCQ7bzB2ZgKdTnbdxNJ7HGs0JdC5ioqM55a5z3mNJALrXX6XWW9T2U9PvSYs8Xniam5sREGC7mB492nKMNTCEyEbOxEQ8k5rQaS8gQPfz0z3ltHFmQlIUVs64v1MqYqQhWPhTLuWsj1L9Vdt81LIcX+TyAMLGxkacPn0aADBkyBCsWLECY8aMQXh4OG6//Xbk5OTgwoULWLNmDQBg9erVmDt3Lt544w2MHz8etbW1WLhwIQICAvDZZ5/JWqaaBhACHLnMkcveWY5aufX9lFy0d+9eCW1XC21emZmZkiRJUmZmpjRq1Cibad544w0pMTFRCgkJkaKioqQnn3xSOn/+vOxlmkwmCYBkMplc7S4ReZg730/eMkFE3SLklgkiou5i4SEi4Vh4iEg4Fh4iEo6Fh4iEY+EhIuFYeIhIOJ/JXPb30aNEWuIThYe5t0TaovlDLebeEmmPpgsPc2+JtEnThYe5t0TapOnCw9xbIm3SdOFh7i2RNmm68DD3lkibNF14mHtLpE2aLjwAc2+JtMgnBhBOSIrCI4mRHLlMpBE+UXiAtsOu5H69vd0NIpJB84daRKQ9LDxEJBwLDxEJx8JDRMKx8BCRcCw8RCQcCw8RCecz43jIdzDG1ve5vMdTWlqK9PR0REdHQ6fTYevWrU6naWlpwe9//3v07dsXer0e8fHxeP/9993pL/m44opajMjfg2kFh/D8hnJMKziEEfl7mCTpY1ze42lqasLgwYMxe/ZsPPHEE7Km+dWvfoX6+nq89957uPPOO1FbWwuLxeJyZ8m3WWNsO+ZFWmNsee+d73C58KSlpSEtLU12++LiYpSUlODMmTMID2+Lp4iPj3d1seTjnMXY6tAWY/tIYiQPu3yAx08uf/DBBxg6dCiWL1+OmJgY9O/fH4sXL8ZPP/1kd5qWlhaYzWabF/k2xtj6F4+fXD5z5gwOHDiA4OBgbNmyBd999x0WLFiA77//HoWFhV1OYzQakZeX5+mukYowxta/eHyPx2KxQKfTYf369Rg2bBgmTpyIFStW4G9/+5vdvZ6cnByYTKb2V01Njae7SV7GGFv/4vHCExUVhZiYGBgMhvb3Bg0aBEmScP78+S6n0ev1CAsLs3mRb2OMrX/xeOFJSUnBxYsX0djY2P7eN998g4CAAMTGxnp68aQRjLH1Ly4XnsbGRpSXl6O8vBwAUFVVhfLycpw7dw5A22FSRkZGe/vp06ejd+/eeOqpp1BZWYnS0lK88MILmD17NkJCQpRZC/IJjLH1I5KL9u7dK6HtIoPNKzMzU5IkScrMzJRGjRplM83XX38tPfzww1JISIgUGxsrLVq0SGpubpa9TJPJJAGQTCaTq90lDbreapEOnv5O2nr8vHTw9HfS9VaLt7tEDrjz/dRJkqT65/uazWYYDAaYTCae7yFSGXe+n7xJlIiEY+EhIuFYeIhIOBYeIhKOhYeIhGPhISLhWHiISDhGn3qJnHhPRoCSr2Lh8YLiilrkba+0yZ+JMgRjSXpi+20BctoQaRVHLgtmL97Tuh+zcsb9AOC0DYsPqYU730/u8QgkJ94z94OvAOgYAUo+jSeXBZIT71lnbkGdmRGg5NtYeARSMraTEaCkZSw8AikZ28kIUNIyFh6B5MR7RobpERnGCFDybSw8AsmJ98ydfDdyJzMClHwbC49gcuI9GQFKvo7jeLyEI5fJV3Acj4b0CNAhuV/vbrch0iIeahGRcCw8RCQcCw8RCcfCQ0TCsfAQkXAsPEQkHAsPEQnHcTykGA54JLlc3uMpLS1Feno6oqOjodPpsHXrVtnTfvrppwgMDMR9993n6mJJ5YorajEifw+mFRzC8xvKMa3gEEbk70FxRa23u0Yq5HLhaWpqwuDBg/HWW2+5NN2PP/6IjIwMPPTQQ64uklTOGufaMeSsznQF89cdY/GhTlw+1EpLS0NaWprLC5o3bx6mT5+OHj16uLSXROomJ86VUa3UkZCTy4WFhThz5gyWLFkiq31LSwvMZrPNi9RJTpwro1qpI48XnlOnTuGll17CunXrEBgobwfLaDTCYDC0v+Li4jzcS3KX3AhWRrXSjTxaeFpbWzF9+nTk5eWhf//+sqfLycmByWRqf9XU1Hiwl9QdciNYGdVKN/Lo5fSGhgYcPXoUx48fR3Z2NgDAYrFAkiQEBgbio48+wtixYztNp9frodfrPdk1Uog1zrXOdKXL8zw6tAWYMaqVbuTRwhMWFoYvv/zS5r23334be/bswaZNm5CQkODJxZMA1jjX+euOQQfYFB9GtZI9LheexsZGnD59uv3nqqoqlJeXIzw8HLfffjtycnJw4cIFrFmzBgEBAUhKSrKZPiIiAsHBwZ3eJ+2yRrV2fORyJB+5THa4XHiOHj2KMWPGtP+8aNEiAEBmZiZWr16N2tpanDt3TrkekiZMSIrCI4mRHLlMsjBzmYi6xZ3vJ28SJSLhWHiISDhN3J1uPRrkCGYi9bF+L105a6OJwtPQ0AAAHMFMpGINDQ0wGAyy2mri5LLFYsHFixcRGhoKnc53rpKYzWbExcWhpqaGJ809hNvYs6zbt7KyEgMGDEBAgLyzN5rY4wkICEBsbKy3u+ExYWFh/FJ4GLexZ8XExMguOgBPLhORF7DwEJFwLDxepNfrsWTJEt4Q60Hcxp7l7vbVxMllIvIt3OMhIuFYeIhIOBYeIhKOhYeIhGPhISLhWHi8IDc3FzqdzuY1cOBAb3fLp1y4cAEzZsxA7969ERISgnvuuQdHjx71drd8Qnx8fKe/X51Oh6ysLNnz0MQtE77o7rvvxscff9z+s9xH/5Bzly9fRkpKCsaMGYNdu3bhtttuw6lTp9CrVy9vd80nHDlyBK2tre0/V1RU4JFHHsGUKVNkz4N/7V4SGBiIyMhIb3fDJ+Xn5yMuLg6FhYXt7/HBAsq57bbbbH5etmwZ+vXrh1GjRsmeBw+1vOTUqVOIjo7GHXfcgSeffJI51Qr64IMPMHToUEyZMgUREREYMmQICgoKvN0tn3T16lWsW7cOs2fPdik5giOXvWDXrl1obGzEgAEDUFtbi7y8PFy4cAEVFRUIDQ31dvc0Lzi47eGBixYtwpQpU3DkyBE8//zzeOedd5CZmenl3vmWf/zjH5g+fTrOnTuH6Oho2dOx8KjAjz/+iL59+2LFihWYM2eOt7ujeUFBQRg6dCgOHjzY/t5zzz2HI0eOoKyszIs98z3jx49HUFAQtm/f7tJ0PNRSgVtuuQX9+/e3eV4ZuS8qKgqJiYk27w0aNIiHswo7e/YsPv74Y/zmN79xeVoWHhVobGzEt99+i6goPvhOCSkpKTh58qTNe9988w369u3rpR75psLCQkRERGDSpEkuT8vC4wWLFy9GSUkJqqurcfDgQTz++OPo0aMHpk2b5u2u+YTf/va3OHToEJYuXYrTp0+jqKgIq1atcmmcCTlmsVhQWFiIzMxM94aCSCTc1KlTpaioKCkoKEiKiYmRpk6dKp0+fdrb3fIp27dvl5KSkiS9Xi8NHDhQWrVqlbe75FM+/PBDCYB08uRJt6bnyWUiEo6HWkQkHAsPEQnHwkNEwrHwEJFwLDxEJBwLDxEJx8JDRMKx8BCRcCw8RCQcCw8RCcfCQ0TC/T8nATXg5J5ufgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 300x300 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(len(iris_species)):\n",
    "    print(f\"{species[i]} correlation visual\")\n",
    "    scatter_line(iris_species[i], 'petal_length', 'petal_width', 3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Data Pre-Processing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "split data into training and test sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#functions required for data-preprocessing and model building\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.decomposition import PCA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "explanatory = iris.drop(columns = ['species']) #explanatory variable\n",
    "response = iris.species #response variable"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "summary statistics; normalization not required"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.057333</td>\n",
       "      <td>3.758000</td>\n",
       "      <td>1.199333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.435866</td>\n",
       "      <td>1.765298</td>\n",
       "      <td>0.762238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       sepal_length  sepal_width  petal_length  petal_width\n",
       "count    150.000000   150.000000    150.000000   150.000000\n",
       "mean       5.843333     3.057333      3.758000     1.199333\n",
       "std        0.828066     0.435866      1.765298     0.762238\n",
       "min        4.300000     2.000000      1.000000     0.100000\n",
       "25%        5.100000     2.800000      1.600000     0.300000\n",
       "50%        5.800000     3.000000      4.350000     1.300000\n",
       "75%        6.400000     3.300000      5.100000     1.800000\n",
       "max        7.900000     4.400000      6.900000     2.500000"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "explanatory.describe() #summary statistics"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Observe the correlation between explanatory varialbes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vzsrICAAEyZMkPf/7rvvkJCQgJEjR+LOnTvYu3cvZs+ejaFDh6otRl7GIiIi0jQl6EGg3bp1w/PnzzF16lTExMSgZs2aOHDggLxoOTo6Gjo6b8dWnJyccPDgQYwePRo1atSAo6MjRo4ciXHjxqktRokgCKI+MZlMhnv37iEuLk5hKAoAGjVqpPT2OpdrLyYMKiabLgcXdwikJEOHhsUdAinBVGpU3CGQkhJS7n6yfaWO6aC2bRv/tlNt2y4uokZ2zp8/j549e+LRo0d4P1eSSCRqvVeeiIiISBmikp3BgwfD09MTe/fuhb29PSQSiarjIiIiooKUoMtYmkBUsnP37l1s3boVFStWVHU8RERERCol6m4sb29v3Lt3T9WxEBERUSEIMkFtizYq9MjOtWvX5F8PHz4c33//PWJiYuDu7i5/kNcbNWrUUF2EREREREVQ6GSnZs2akEgkCgXJAwYMkH/95j0WKBMREamZlo7AqEuhk50HDx6oMw4iIiIitSh0slOuXDn51ydPnoSvry/09BRXz8nJwdmzZxX6EhERkYq9N78dfZioAuUmTZogISEhT/vLly/RpEmTIgdFREREpCqibj1/U5vzvhcvXqB06dJFDoqIiIg+gDU7SlEq2enUqROA18XI/fr1U3gKam5uLq5duwZfX1/VRkhERESKmOwoRalkx8zMDMDrkR0TExMYGhrK39PX10e9evUwaNAg1UZIREREVARKJTurV68GALi4uGDMmDG8ZEVERFQMRD7D+7MlqmZn2rRpqo6DiIiISC0KnezUqlWr0A/8vHLliuiAiIiI6CNYs6OUQic7/v7+8q9fvXqFRYsWoVq1avDx8QEAnD9/Hjdv3sSQIUNUHiQRERGRWIVOdt69dDVw4ECMGDECP/74Y54+jx8/Vl10RERElBdHdpQialLBf/75BwEBAXnae/fujW3bthU5KCIiIiJVEZXsGBoa4syZM3naz5w5AwMDgyIHRURERAUTZILaFm0k6m6sUaNG4bvvvsOVK1fg5eUFAAgNDcWqVaswZcoUlQZIRERE79HSpERdRCU748ePR/ny5fHHH39g/fr1AAA3NzesXr0aXbt2VWmAREREREUhKtkBgK5duzKxISIiKg586LlSRNXsEBEREWmKQo/sWFpa4s6dO7CysoKFhcUHJxhMSEhQSXBERESUl7YWEqtL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lC//8eeF8NlYmoXPxtI8fDaWZuGzsTTPp3w2VnXbemrb9o3Y/G+i0WQc2SEiItIw2lpboy5qLVAmIiIiKm5qHdlp2LAhDA0N1bkLIiKizw5rdpRT6GQnOTm50Bs1NTUFAOzbt0/5iIiIiOiDeBlLOYVOdszNzRXuNMqPIAiQSCTIzc0tcmBEREREqlDoZOfYsfyfOkxERESfFi9jKafQyU5hHvBJREREVNIUqUA5PT0d0dHRyMrKUmivUaNGkYIiIiKigrFmRzmikp3nz5+jf//+2L9/f77vs2aHiIiISgpR8+yMGjUKSUlJCA0NhaGhIQ4cOIC1a9eiUqVK2LVrl6pjJCIionfIBEFtizYSNbJz9OhR7Ny5E56entDR0UG5cuXQokULmJqaIigoCG3btlV1nERERESiiBrZSUtLg43N66fUWlhY4Pnz5wAAd3d3XLlyRXXRERERUR6CGv9pI1HJTpUqVRAZGQkA8PDwwNKlS/HkyRMsWbIE9vb2Kg2QiIiIFAmCTG2LNhJ1GWvkyJF49uwZAGDatGlo3bo1NmzYAH19faxZs0aV8REREREViahkp3fv3vKv69Spg0ePHiEiIgLOzs6wsrJSWXBERESUl0xLLzepi6jLWDNnzkR6err8tZGREWrXro3SpUtj5syZKguOiIiIqKhEJTszZsxAampqnvb09HTMmDGjyEERERFRwQRBUNuijUQlO28e+Pm+q1evwtLSsshBEREREamKUjU7FhYWkEgkkEgkqFy5skLCk5ubi9TUVAwePFjlQRIREdFbrNlRjlLJTnBwMARBwIABAzBjxgyYmZnJ39PX14eLiwt8fHxUHiQRERGRWEolO3379gUAuLq6on79+tDTK9JzRImIiEgEba2tURdRNTt+fn549OgRJk+ejB49eiAuLg4AsH//fty8eVOlARIREZEiPhtLOaKSnRMnTsDd3R2hoaHYvn27/M6sq1evYtq0aSoNkIiIiKgoRCU748ePx08//YTDhw9DX19f3t60aVOcP39eZcERERFRXnw2lnJEJTvXr19Hx44d87Tb2NggPj6+yEERERERqYqoZMfc3Fz+bKx3hYWFwdHRschBERERUcE4qaByRCU73bt3x7hx4xATEwOJRAKZTIYzZ85gzJgxCAgIUHWMRERERKKJSnZmz56NqlWrwsnJCampqahWrRoa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",
      "text/plain": [
       "<Figure size 700x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,6))\n",
    "sns.heatmap(explanatory.corr(), annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Perform PCA to handle multicollinearity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PC1</th>\n",
       "      <th>PC2</th>\n",
       "      <th>PC3</th>\n",
       "      <th>PC4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-2.684126</td>\n",
       "      <td>0.319397</td>\n",
       "      <td>-0.027915</td>\n",
       "      <td>-0.002262</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-2.714142</td>\n",
       "      <td>-0.177001</td>\n",
       "      <td>-0.210464</td>\n",
       "      <td>-0.099027</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-2.888991</td>\n",
       "      <td>-0.144949</td>\n",
       "      <td>0.017900</td>\n",
       "      <td>-0.019968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-2.745343</td>\n",
       "      <td>-0.318299</td>\n",
       "      <td>0.031559</td>\n",
       "      <td>0.075576</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-2.728717</td>\n",
       "      <td>0.326755</td>\n",
       "      <td>0.090079</td>\n",
       "      <td>0.061259</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        PC1       PC2       PC3       PC4\n",
       "0 -2.684126  0.319397 -0.027915 -0.002262\n",
       "1 -2.714142 -0.177001 -0.210464 -0.099027\n",
       "2 -2.888991 -0.144949  0.017900 -0.019968\n",
       "3 -2.745343 -0.318299  0.031559  0.075576\n",
       "4 -2.728717  0.326755  0.090079  0.061259"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pca = PCA(random_state = 2000, n_components = None)\n",
    "explanatory_pca = pca.fit_transform(explanatory) # apply PCA to explanatory\n",
    "\n",
    "explanatory_pca = pd.DataFrame(explanatory_pca) \n",
    "explanatory_pca = explanatory_pca.rename(columns={0: \"PC1\", 1: \"PC2\", 2: \"PC3\", 3: \"PC4\"}) # set the column names\n",
    "\n",
    "explanatory_pca.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 700x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(7,6))\n",
    "sns.heatmap(explanatory_pca.corr(),annot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Prepare for the model training; split data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = explanatory_pca\n",
    "y = response\n",
    "\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Modling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-2\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" checked><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">DecisionTreeClassifier</label><div class=\"sk-toggleable__content\"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = DecisionTreeClassifier() #define the model\n",
    "\n",
    "model.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Make Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train_pred = model.predict(x_train)\n",
    "y_test_pred = model.predict(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Evaluating"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>140</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>83</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>126</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>85</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>131</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>54</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>120</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>124</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>118</th>\n",
       "      <td>virginica</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>136</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>49</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>104</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>117</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>132</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>59</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>105</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>130</th>\n",
       "      <td>virginica</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>82</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>99</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>87</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>66</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Actual   Predicted     result\n",
       "140   virginica   virginica    Correct\n",
       "83   versicolor   virginica  Incorrect\n",
       "74   versicolor  versicolor    Correct\n",
       "126   virginica   virginica    Correct\n",
       "34       setosa      setosa    Correct\n",
       "85   versicolor  versicolor    Correct\n",
       "108   virginica   virginica    Correct\n",
       "47       setosa      setosa    Correct\n",
       "24       setosa      setosa    Correct\n",
       "131   virginica   virginica    Correct\n",
       "54   versicolor  versicolor    Correct\n",
       "120   virginica   virginica    Correct\n",
       "124   virginica   virginica    Correct\n",
       "109   virginica   virginica    Correct\n",
       "40       setosa      setosa    Correct\n",
       "51   versicolor  versicolor    Correct\n",
       "29       setosa      setosa    Correct\n",
       "48       setosa      setosa    Correct\n",
       "118   virginica  versicolor  Incorrect\n",
       "136   virginica   virginica    Correct\n",
       "21       setosa      setosa    Correct\n",
       "27       setosa      setosa    Correct\n",
       "32       setosa      setosa    Correct\n",
       "44       setosa      setosa    Correct\n",
       "49       setosa      setosa    Correct\n",
       "104   virginica   virginica    Correct\n",
       "117   virginica   virginica    Correct\n",
       "11       setosa      setosa    Correct\n",
       "132   virginica   virginica    Correct\n",
       "59   versicolor  versicolor    Correct\n",
       "105   virginica   virginica    Correct\n",
       "33       setosa      setosa    Correct\n",
       "14       setosa      setosa    Correct\n",
       "36       setosa      setosa    Correct\n",
       "130   virginica  versicolor  Incorrect\n",
       "82   versicolor  versicolor    Correct\n",
       "86   versicolor   virginica  Incorrect\n",
       "99   versicolor  versicolor    Correct\n",
       "62   versicolor  versicolor    Correct\n",
       "87   versicolor  versicolor    Correct\n",
       "67   versicolor  versicolor    Correct\n",
       "66   versicolor  versicolor    Correct\n",
       "60   versicolor  versicolor    Correct\n",
       "52   versicolor   virginica  Incorrect\n",
       "148   virginica   virginica    Correct"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})\n",
    "result[\"result\"] = np.where(result.Actual == result.Predicted, \"Correct\", \"Incorrect\")\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>83</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>118</th>\n",
       "      <td>virginica</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>130</th>\n",
       "      <td>virginica</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>86</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>52</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Incorrect</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Actual   Predicted     result\n",
       "83   versicolor   virginica  Incorrect\n",
       "118   virginica  versicolor  Incorrect\n",
       "130   virginica  versicolor  Incorrect\n",
       "86   versicolor   virginica  Incorrect\n",
       "52   versicolor   virginica  Incorrect"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result.loc[result.result == \"Incorrect\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Correct      40\n",
       "Incorrect     5\n",
       "Name: result, dtype: int64"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result[\"result\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Accuracy: 1.000\n",
      "Testing Accuracy: 0.889\n",
      "[[35  0  0]\n",
      " [ 0 35  0]\n",
      " [ 0  0 35]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "      setosa       1.00      1.00      1.00        35\n",
      "  versicolor       1.00      1.00      1.00        35\n",
      "   virginica       1.00      1.00      1.00        35\n",
      "\n",
      "    accuracy                           1.00       105\n",
      "   macro avg       1.00      1.00      1.00       105\n",
      "weighted avg       1.00      1.00      1.00       105\n",
      "\n",
      "[[15  0  0]\n",
      " [ 0 12  3]\n",
      " [ 0  2 13]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "      setosa       1.00      1.00      1.00        15\n",
      "  versicolor       0.86      0.80      0.83        15\n",
      "   virginica       0.81      0.87      0.84        15\n",
      "\n",
      "    accuracy                           0.89        45\n",
      "   macro avg       0.89      0.89      0.89        45\n",
      "weighted avg       0.89      0.89      0.89        45\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score, r2_score\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "#Evaluation\n",
    "print('Training Accuracy: {:.3f}'.format(accuracy_score(y_train, y_train_pred)))\n",
    "print('Testing Accuracy: {:.3f}'.format(accuracy_score(y_test, y_test_pred)))\n",
    "\n",
    "print(confusion_matrix(y_train, y_train_pred))\n",
    "print(classification_report(y_train, y_train_pred))\n",
    "\n",
    "print(confusion_matrix(y_test, y_test_pred))\n",
    "print(classification_report(y_test, y_test_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Conclusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "class classifier:\n",
    "    def __init__(self, data):\n",
    "        self.answer = pd.DataFrame(data[\"species\"])\n",
    "        data = data.drop(columns = [\"species\"])\n",
    "        \n",
    "        self.data_pca = pca.transform(data)\n",
    "        self.data_pca = pd.DataFrame(self.data_pca)\n",
    "    \n",
    "    def predict(self):\n",
    "        return model.predict(self.data_pca)\n",
    "    \n",
    "    def accuracy(self):\n",
    "        return accuracy_score(self.answer, self.predict())\n",
    "    \n",
    "    def report(self):\n",
    "        dic = {\"Actual\": self.answer[\"species\"], \"Predicted\": self.predict()}\n",
    "        result = pd.DataFrame(dic)\n",
    "        result[\"result\"] = np.where(result.Actual == result.Predicted, \"Correct\", \"Incorrect\")\n",
    "\n",
    "        print(\"Accuracy:\", self.accuracy())\n",
    "        return result\n",
    "    \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal_length</th>\n",
       "      <th>sepal_width</th>\n",
       "      <th>petal_length</th>\n",
       "      <th>petal_width</th>\n",
       "      <th>species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.0</td>\n",
       "      <td>3.2</td>\n",
       "      <td>4.7</td>\n",
       "      <td>1.4</td>\n",
       "      <td>versicolor</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6.3</td>\n",
       "      <td>3.3</td>\n",
       "      <td>6.0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   sepal_length  sepal_width  petal_length  petal_width     species\n",
       "0           5.1          3.5           1.4          0.2      setosa\n",
       "1           7.0          3.2           4.7          1.4  versicolor\n",
       "2           6.3          3.3           6.0          2.5   virginica"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# select three rows from each species\n",
    "test_data = sns.load_dataset('iris')\n",
    "\n",
    "sesota = test_data[test_data[\"species\"] == 'setosa'].head(1)\n",
    "versicolor = test_data[test_data[\"species\"] == 'versicolor'].head(1)\n",
    "virginica = test_data[test_data[\"species\"] == 'virginica'].head(1)\n",
    "\n",
    "\n",
    "combined_data = pd.concat([sesota, versicolor, virginica], ignore_index=True)\n",
    "combined_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "# select 10 random rows\n",
    "\n",
    "random_data = test_data.sample(n = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 1.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\ybrot\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
      "  warnings.warn(\n",
      "c:\\Users\\ybrot\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Actual   Predicted   result\n",
       "0      setosa      setosa  Correct\n",
       "1  versicolor  versicolor  Correct\n",
       "2   virginica   virginica  Correct"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = classifier(combined_data)\n",
    "\n",
    "result.report()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 1.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\ybrot\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
      "  warnings.warn(\n",
      "c:\\Users\\ybrot\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\sklearn\\base.py:439: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>Predicted</th>\n",
       "      <th>result</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>88</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>56</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>131</th>\n",
       "      <td>virginica</td>\n",
       "      <td>virginica</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>51</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>47</th>\n",
       "      <td>setosa</td>\n",
       "      <td>setosa</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>90</th>\n",
       "      <td>versicolor</td>\n",
       "      <td>versicolor</td>\n",
       "      <td>Correct</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Actual   Predicted   result\n",
       "3        setosa      setosa  Correct\n",
       "64   versicolor  versicolor  Correct\n",
       "88   versicolor  versicolor  Correct\n",
       "69   versicolor  versicolor  Correct\n",
       "56   versicolor  versicolor  Correct\n",
       "131   virginica   virginica  Correct\n",
       "51   versicolor  versicolor  Correct\n",
       "60   versicolor  versicolor  Correct\n",
       "47       setosa      setosa  Correct\n",
       "90   versicolor  versicolor  Correct"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = classifier(random_data)\n",
    "\n",
    "result.report()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
