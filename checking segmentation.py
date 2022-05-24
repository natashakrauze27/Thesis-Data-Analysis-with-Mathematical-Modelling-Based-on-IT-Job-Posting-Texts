#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statistics
from scipy.stats import norm, kstest


# # ANOVA TESTS

# The assumptions for implementing one way ANOVA include:
# - The normality criterion: each group compared should come from a population following the normal distribution.
# - The variance criterion (or 'homogeneity of variances'): samples should come from populations with the same variance.
# - Independent samples: performance (the dependent variable) in each sample should not be affected by the conditions in other samples.

# # Average Check ANOVA test

# In[338]:


df1 = pd.read_csv('АноваСредниеЗнач.csv', sep=';', encoding='utf-8-sig') 


# In[339]:


df1.describe()


# ### checking test criterions:

# In[340]:


Cluster_1 = (df1['Кластер 1']).dropna()
Cluster_2 = (df1['Кластер 2']).dropna()
Cluster_3 = (df1['Кластер 3']).dropna()
Cluster_4 = (df1['Кластер 4']).dropna()
Cluster_5 = (df1['Кластер 5']).dropna()


# In[341]:


Cluster_1


# In[296]:


stat, p = stats.normaltest(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[297]:


result = stats.anderson(Cluster_1)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[298]:


stat, p = stats.shapiro(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# In[299]:


print(stats.kstest(Cluster_1, 'norm'))
print(stats.kstest(Cluster_2, 'norm'))
print(stats.kstest(Cluster_3, 'norm'))
print(stats.kstest(Cluster_4, 'norm'))


# Sample does not look Gaussian (reject H0)

# ### computing histograms for data

# In[301]:


df1.hist(column='Кластер 1')


# In[302]:


df1.hist(column='Кластер 2')


# In[303]:


df1.hist(column='Кластер 3')


# In[304]:


df1.hist(column='Кластер 4')


# ### One-way ANOVA TEST

# In[305]:


F, p = stats.f_oneway(Cluster_1, Cluster_2, Cluster_3, Cluster_4)
print('F statistic = {:5.3f} and probability p = {:5.3f}'.format(F, p)) 


# As p > a (0.05) we state that we do not have a main interaction effect. This simply means that amongst group comparison identifies statistically insignificant differences. 

# # Average Number of goods ANOVA test

# In[280]:


df2 = pd.read_csv('АноваКолТоваров.csv', sep=';', encoding='utf-8-sig') 


# In[281]:


df2.describe()


# In[282]:


Cluster_1 = (df2['Кластер 1']).dropna()
Cluster_2 = (df2['Кластер 2']).dropna()
Cluster_3 = (df2['Кластер 3']).dropna()
Cluster_4 = (df2['Кластер 4']).dropna()


# ### checking test criterions:

# In[288]:


stat, p = stats.normaltest(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[290]:


result = stats.anderson(Cluster_1)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[291]:


stat, p = stats.shapiro(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# In[277]:


print(stats.kstest(Cluster_1, 'norm'))
print(stats.kstest(Cluster_2, 'norm'))
print(stats.kstest(Cluster_3, 'norm'))
print(stats.kstest(Cluster_4, 'norm'))


# Sample does not look Gaussian (reject H0)

# ### computing histograms for data

# In[278]:


df2.hist(column='Кластер 1')


# In[279]:


df2.hist(column='Кластер 2')


# In[216]:


df2.hist(column='Кластер 3')


# In[217]:


df2.hist(column='Кластер 4')


# ### One-way ANOVA TEST

# In[221]:


F, p = stats.f_oneway(Cluster_1, Cluster_2, Cluster_3, Cluster_4)
print('F statistic = {:5.3f} and probability p = {:5.3f}'.format(F, p)) 


# As p < a (0.05) we state that we have a main interaction effect. This simply means that amongst group comparison identifies statistically significant differences. 

# # Average Revenue ANOVA test
# 

# In[309]:


df3 = pd.read_csv('АноваВыручка.csv', sep=';', encoding='utf-8-sig') 


# In[310]:


Cluster_1 = (df3['Кластер 1']).dropna()
Cluster_2 = (df3['Кластер 2']).dropna()
Cluster_3 = (df3['Кластер 3']).dropna()
Cluster_4 = (df3['Кластер 4']).dropna()


# In[311]:


df3.describe()


# ### checking test criterions:

# In[313]:


stat, p = stats.normaltest(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[314]:


result = stats.anderson(Cluster_1)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[315]:


stat, p = stats.shapiro(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# In[316]:


print(stats.kstest(Cluster_1, 'norm'))
print(stats.kstest(Cluster_2, 'norm'))
print(stats.kstest(Cluster_3, 'norm'))
print(stats.kstest(Cluster_4, 'norm'))


# Sample does not look Gaussian (reject H0)

# ### computing histograms for data

# In[317]:


df3.hist(column='Кластер 1')


# In[318]:


df3.hist(column='Кластер 2')


# In[319]:


df3.hist(column='Кластер 3')


# In[320]:


df3.hist(column='Кластер 4')


# ### one-way ANOVA test

# In[322]:


F, p = stats.f_oneway(Cluster_1, Cluster_2, Cluster_3, Cluster_4)
print('F statistic = {:5.3f} and probability p = {:5.3f}'.format(F, p)) 


# As p < a (0.05) we state that we have a main interaction effect. This simply means that amongst group comparison identifies statistically significant differences. 

# # Average Margin ANOVA test
# 
# 

# In[325]:


df4 = pd.read_csv('АноваМаржа.csv', sep=';', encoding='utf-8-sig') 


# In[326]:


Cluster_1 = (df4['Кластер 1']).dropna()
Cluster_2 = (df4['Кластер 2']).dropna()
Cluster_3 = (df4['Кластер 3']).dropna()
Cluster_4 = (df4['Кластер 4']).dropna()


# In[327]:


df4.describe()


# ### checking test criterions:
# 

# In[328]:


stat, p = stats.normaltest(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[329]:


result = stats.anderson(Cluster_1)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))


# In[330]:


stat, p = stats.shapiro(Cluster_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')


# In[331]:


print(stats.kstest(Cluster_1, 'norm'))
print(stats.kstest(Cluster_2, 'norm'))
print(stats.kstest(Cluster_3, 'norm'))
print(stats.kstest(Cluster_4, 'norm'))


# Sample does not look Gaussian (reject H0)

# ### computing histograms for data
# 

# In[332]:


df4.hist(column='Кластер 1')


# In[333]:


df4.hist(column='Кластер 2')


# In[334]:


df4.hist(column='Кластер 3')


# In[335]:


df4.hist(column='Кластер 4')


# ### one-way ANOVA test

# In[336]:


F, p = stats.f_oneway(Cluster_1, Cluster_2, Cluster_3, Cluster_4)
print('F statistic = {:5.3f} and probability p = {:5.3f}'.format(F, p)) 


# As p < a (0.05) we state that we have a main interaction effect. This simply means that amongst group comparison identifies statistically significant differences. 
