## About the repository
This is a repository of all the Python code used for my Master's thesis at KU Leuven in Statistics,
titled **"Modeling of protein and dry matter content using NIR data of insect feed and products"**. It also includes the
thesis itself. The novelty lies in the fact that it is entirely in Python while most of the chemometric techniques are only available in Matlab
through use of commercial tools (PLS_Toolbox) or sometimes in R. Indeed, the chemometrics course at KU Leuven uses Solo, an expensive GUI statistical software,
which implements PLS_Toolbox without Matlab. While my implementations are very far from optimal, most of them had to be self implemented from zero, based on algorithms outlined
in old papers (often not very clearly) and was only used on a relatively small dataset, so efficiency was not a concern. Also, I don't think anyone uses big data in the field, so unless someone wants to implement these techniques in some embedded system, efficiency is irrelevant. I hope my work might save a lot of headache for someone trying to do chemometrics in Python or without paying licensing fees. At least at the sponsoring company, Protix.  

**Unfortunately, the underlying dataset could not be made public.** In addition, NIR datasets are not widely available in general, especially not with measurement day information. I might include an example dataset at a later time.

># Used techniques
> - Measurement Day Cross-Validation
> - Conversion to absorbance
> - Baseline correction and detrending
> - Savitzky-Golay Smoothing and derivatives
> - Multiplicative Scatter Correction
> - Orthogonal Signal Correction
> - Standard Normal Variate
> - Partial Least Squares
> - Interval Partial Least Squares
> - Maximum Correntropy Weighted Partial Least Squares (courtesy of Valeria Fonseca Diaz)
> - Support Vector Regression
> - Stepwise multiple regression
> - Bunch of plotting tools
