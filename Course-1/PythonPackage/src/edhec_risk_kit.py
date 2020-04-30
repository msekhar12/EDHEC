# edhec_risk_kit.py

import pandas as pd
import numpy as np
import os
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


def read_ind_returns(data_loc='../data'):
    """
    Reads the data from ind30_m_vw_rets.csv and prepares the data for analysis
    """
    # Prepare the file location
    file_loc = os.path.join(data_loc, 'ind30_m_vw_rets.csv')

    # Divide the data with 100, to convert that from percentage to proportion
    ind = pd.read_csv(file_loc,
                      header=0,
                      index_col=0,
                      na_values=-99.99,
                      parse_dates=True)/100

   # Format the index to a proper date format 'YYYY-MM'
    ind.index = pd.to_datetime(ind.index,
                               format='%Y%m').to_period('M')

    # Remove the extra space from each column name
    ind.columns = ind.columns.str.strip()

    return ind


def read_Portfolios_Formed_on_ME_monthly_EW(data_loc='../data'):
    '''
    Reads the data from Portfolios_Formed_on_ME_monthly_EW.csv and
    prepares the data for analysis
    '''
    file_loc = os.path.join(data_loc, 'Portfolios_Formed_on_ME_monthly_EW.csv')
    me_m = pd.read_csv(file_loc,
                       header=0,
                       index_col=0,
                       na_values=-99.99,
                       parse_dates=True)

    me_m = me_m/100
    me_m.index = pd.to_datetime(me_m.index, format='%Y%m')
    me_m.index = me_m.index.to_period('M')
    return me_m


def read_edhec_hedgefundindices(data_loc='../data'):
    '''
    Reads the data from edhec-hedgefundindices.csv and prepares the data
    for analysis
    '''
    file_loc = os.path.join(data_loc, 'edhec-hedgefundindices.csv')
    hfi = pd.read_csv(file_loc,
                      header=0,
                      index_col=0,
                      parse_dates=True)

    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def annualized_returns(r, time_unit='month'):
    '''
    Function to compute the annualized returns.
    Inputs:
        r: Series or DataFrame
        time_periods: Can be 'month' or 'day' or 'year'
    Output:
        Annualized return. If input is a Series, then float, else a Series.
        None if the input is empty or None
    '''
    assert(time_unit in ('month', 'year', 'day'))

    total_periods = r.shape[0]
    if (total_periods == 0):
        return None

    # geometric mean of the data
    gm = ((r + 1).prod())**(1/total_periods)

    if time_unit == 'day':
        n = 252
    elif time_unit == 'month':
        n = 12
    elif time_unit == 'year':
        n = 1
    return gm**n - 1


def annualized_volatility(r, time_unit='month'):
    '''
    Function to compute the annualized volatility.
    Inputs:
        r: Series or DataFrame
        time_periods: Can be 'month' or 'day' or 'year'
    Output:
        Annualized return. If input is a Series, then float, else a Series.
        None if the input is empty or None
    '''
    assert(time_unit in ('month', 'year', 'day'))

    total_periods = r.shape[0]

    if (total_periods == 0):
        return None

    if time_unit == 'month':
        return r.std() * np.sqrt(12)
    elif time_unit == 'day':
        return r.std() * np.sqrt(252)
    elif time_unit == 'year':
        return r.std()


def annualized_risk_free_return(risk_free_return, time_unit='month'):
    '''
    Input:
        risk_free_return: Risk Free rate (floatpointing number between [0,1])
        time_unit: Time unit of the risk free rate
    Output:
        Annualized risk free rate
    '''
    assert(time_unit in ('month', 'year', 'day'))

    if time_unit == 'month':
        return (risk_free_return + 1)**12 - 1
    elif time_unit == 'day':
        return (risk_free_return + 1)**252 - 1
    elif time_unit == 'year':
        return risk_free_return


def sharpe_ratio(r, risk_free_return, r_time_unit='month',
                 risk_free_return_time_unit='year'):
    '''
    Computes the annulized sharpe ratio.
    Logic:
        1. Annualize the Risk Free rate (call this as rf_annualized)
        2. Based on the r_time_unit, convert the rf_annualized to Risk Free per
           unit of r_time_unit (Call this as rf)
        3. Subtract the rf from r. (Call this as excess_return)
        4. Get annualized returns of excess_return
           (call this as annualized_excess_return)
        5. Get annualized volatility of excess_return
           (call this as annualized_excess_volatility)
        6. Divide annualized_excess_return by annualized_excess_volatility

    Inputs:
        r: Can be a Series or a DataFrame
        riskfree_return: floating point value in the range [0,1]
        r_time_unit: r's time unit
        riskfree_return_time_uint: riskfree_return's time unit
    Output:
        Sharpe Ratio (float)
    '''

    # Find the risk free annualized return
    rf_annualized = annualized_risk_free_return(
        risk_free_return=risk_free_return,
        time_unit=risk_free_return_time_unit)

    # Now convert the annualized risk free rate to the rate per r_time_unit

    if r_time_unit == 'month':
        rf = (rf_annualized+1) ** (1/12) - 1
    elif r_time_unit == 'day':
        rf = (rf_annualized+1) ** (1/252) - 1
    elif r_time_unit == 'year':
        rf = rf_annualized

    # Find the excess return
    excess_return = r - rf

    # Find the annualized excess return
    annualized_excess_return = annualized_returns(r=excess_return,
                                                  time_unit=r_time_unit)

    # Find the annualized volatility
    annualized_excess_volatility = annualized_volatility(r=excess_return,
                                                         time_unit=r_time_unit)

    # Find Sharpe Ratio
    return annualized_excess_return/annualized_excess_volatility

# Max drawdown


def drawdown(S):
    '''
    Takes a Pandas Time Series of returns and finds a DataFrame with 4 columns:
    Input Series
    Wealth Index (assuming that you have invested $1000)
    Previous peak
    Drawdown
    '''
    # Find the Wealth_index
    wealth_index = ((S+1).cumprod()) * 1000

    # Find the previous peak
    previous_peak = wealth_index.cummax()

    # Drawdown
    drawdown = wealth_index/previous_peak - 1

    return pd.DataFrame({'Returns': S,
                         'Wealth_Index': wealth_index,
                         'Previous_Peak': previous_peak,
                         'Drawdown': drawdown})


def skewness(df):
    '''
    Finds the skewness
    Accepts a Series or a Data Frame as input and computes the skewness of
    the data.
    Returns a Pandas Series with index as the column names of the input data
    frame.
    Returns a float value if the input is a Series object.
    '''
    return ((df - df.mean())**3).mean()/(df.std(ddof=0)**3)


def kurtosis(df):
    '''
    Finds the kurtosis
    Accepts a Series or a Data Frame as input and computes the skewness of the
    data.
    Returns a Pandas Series with the index as the column names of the input
    data frame.
    Returns a float value if the input is a Series object.
    '''
    return ((df - df.mean())**4).mean()/(df.std(ddof=0)**4)


def is_normal(r, level=0.01):
    '''
    Input: Series
    Output: If the distribution is Normal using the Jarque-Bera output and
    Chi-square test, then [True, p_value] else [False, p_value]

    Null Hypothesis: The distribution is Normal
    Alt Hypothesis: The distribution is NOT Normal

    If the p-value obtained is <= level (or alpha), then we reject
    the null hypothesis, as the probability that pure randomness has
    caused us to get the sample data is lesser than the significance level
    '''

    statistic, p_value = scipy.stats.jarque_bera(r)
    if p_value > level:
        # i.e, not able to reject null hypothesis
        return [True, p_value]
    else:
        # i.e, reject the null hypothesis
        return [False, p_value]


def jarque_bera_test(df, alpha=0.01):
    '''
    Input can be a Data Frame or a Series, and alpha
    Output is True/False.
    True, if the distribution is Normal at the given significance level (aplha)
            else False
    '''
    return df.aggregate(is_normal, level=alpha)


def semi_deviation(df):
    '''
    Calculates the std. deviation for the data confining the data to only the
    observations that are < mean of the data (but we will use 0 instead of mean)
    For high frequency returns, the avg is close to 0, so we can use that

    Input: Data Frame or Series
    Output: Std. Dev of the lower returns (below avg returns)
    '''
    return df[df < 0].std(ddof=0)


def var_historic(r, level=0.01):
    '''
    Returns the historical value at risk as a specified significance level i.e
    returns the number such that "level" percent of returns fall below that
    number, and the (100 - level) percent are above!!

    Input:
        r: Series or DataFrame
        level: Alpha or significance level
    Output:
        Value at Risk using the historical method
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic,
                           level=level)
        # we are applying var_historic() on each column of DataFrame and note
        # that its a recursive function
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def cvar_historic(r, level=0.01):
    '''
    Computes the conditional VaR of a Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        VaR = -var_historic(r, level=level)
        return -r[r <= VaR].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")


def var_gaussian(r, level=0.01, modified=False):
    '''
    Returns parametric gaussian VaR of a Series or DataFrame.
    If modified is True, then the modified Var is returned using Corning-Fisher
    modification
    Input:
        r : Can be a Data Frame or a Series
        level: Significance level or alpha
        modified: If True, then use corning-fisher method, else Gaussian
    Output:
        Series: if the input is a Data Frame
        float: if the input is a Series
    '''

    # Get the z-score, for a given alpha level
    z = norm.ppf(level)

    if modified:
        # Modify the z-score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)

        z = (z + (z**2 - 1)*s/6 +
             (z**3 - 3*z) * (k-3)/24 -
             (2*z**3 - 5*z) * (s**2)/36)
    return -(r.mean() + z*r.std(ddof=0))


def portfolio_return(weights, returns):
    '''
    Takes a weights vector and expected returns vector and returns expected returns
    Formula: transpose(W) X returns
    Input:
        weights: numpy array
        returns: pandas Series
    Output:
        a floating point number showing the portfolio return
    '''
    return weights.T @ returns


def portfolio_volatility(weights, cov):
    '''
    Takes a weight vector and covariance as inputs and returns std. dev of portfolio
    Formula: sqrt(transpose(w) X cov X w)
    Inputs:
        weights: numpy array
        cov: A covariance matrix (pandas Data Frame)
    Output:
        floating point number showing the volatility of the portfolio
    '''

    return (weights.T @ cov @ weights)**0.5


def plot_ef2(n_points, er, cov, style=".-"):
    '''
    Plots efficient frontier for 2 assets
    Inputs:
        n_points: Number of different pairs of weights to try
        er: expected return vector (Series)
        cov: Pandas DataFrame containing the covariance of the assets
        style: For plotting
    '''

    # Generate the n_points of pairs of weights
    # Each pair must sum to 1
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]

    # Get the portfolio expected return for each pair of weights
    rets = [portfolio_return(returns=er, weights=w) for w in weights]

    # Get the portfolio volatility for each pair of weights
    vol = [portfolio_volatility(cov=cov, weights=w) for w in weights]

    df = pd.DataFrame({"Returns": rets,
                       "Volatility": vol})

    return df.plot.line(x="Volatility", y="Returns", style=style)


# Minimize volatility function
def minimize_volatility(target_return, er, cov):
    '''
    For a target_return, get the weights that have the minimum volatility
    Inputs:
        target_return: Floating point number representing the desired target retrun
        er: Expected returns of all assets. A pandas Series
        cov: A pandas data frame representing the covariance matrix of the assets
    '''

    # Get the number of assets we have:
    n = er.shape[0]

    # For the optimizer to work, we need to give it
    # an objective function (which needs to be optimized)
    # and a set of constraints.
    # We need to provide an initial guess of weights.
    # Some people put all the money in one asset as the initial
    # guess. But we will put equal weights for all the assets
    # as an initial guess.

    init_guess = np.repeat(1/n, n)

    # Let us provide the constraints

    # Constraint-1: Provide bounds for weights
    # -------------
    # Each weight must be confined to the interval: [0,1]
    # If the weight is negative, then it is equivalent to
    # going short.
    # If the weight is beyond 1, then it is equivalent to
    # leverage.
    # Hence the weight must be within the interval: [0,1]
    # You have to provide a sequence of bounds for every weight.
    # Note: In the below statement, the extra comma(,) is necessary,
    # as if we just include ((0,1)), then the outer parenthesis
    # will be treated as normal parenthesis in a mathematical expression
    # and hence ((0,1)) is equivalent to (0,1)
    # ((0,1),) will be equal to a tuple with one element (0,1)

    bounds = ((0.0, 1.0),) * n

    # Constraint-2:
    # -------------
    # The weights when matrix multiplied with expected returns must
    # give us desired returns.
    #
    # That is transpose(weights) X Expected_Returns = Desired_Return
    # Here the weights are the ones which are found by the optimizer,
    # and those determined weights must satisfy this constraint
    # The desired weight is nothing but target_return parameter
    # This constraint is an equality constraint.
    # The minimize() function of scipy.optimize works as follows:
    #   For each set of weights the optimizer generates, it will
    #   check whether this constraint function returns 0 or not
    #   If 0 is returned then the constraint is said to be satisfied,
    #   else it is dissatisfied.
    #   Therefore write a function that returns the following:
    #   target_return - Weights.T @ er

    # In the below statement, 'args' will take all the arguments
    # that are needed for the 'fun', except the first parameter
    # (weights in this example). The first parameter is the one
    # that we optimize
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }

    # Constraint-3:
    # -------------
    # Sum of all weights must sum to 1
    # Similar to constraint-2, the function must return 0
    # when the constraint is satisfied.

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # We defined 3 constraints:
    # Constraint-1: For bounds on weights
    # Constraint-2: Calculated return using the
    #               weights is the target return
    # Constraint-3: Weights sum to 1

    # Now we are ready to call the optimizer
    # We already imported the optimizer as
    # from scipy.optimize import minimize

    # The minimize() function will take the objective function
    # which is nothing but the portfolio volatility
    # We already written a function portfolio_volatility(weights, cov)
    # The weights are the ones the function will optimize.
    # The initial weights are assigned to the init_weights
    # The args (except the weights arg, which must be the first argument of
    # portfolio_volatility) will be supplied using the 'args' option
    # 'args' = (cov)
    # We will use 'SLSQP' as the quadratic programming optimizer
    # method = 'SLSQP'
    # You need to also supply the constraints and bounds we created.
    # The minimize() function will produce a lot of informational
    # messages. So supress those messages using disp

    results = minimize(portfolio_volatility,
                       init_guess,
                       args=(cov),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target,
                                    weights_sum_to_1),
                       bounds=bounds)

    # The optimized weigts are present in results.x
    return results.x


def optimal_weights(n_points, er, cov):
    '''
    Generated a tuple of optimal weights for each desired expected return
    Input:
        n_points: Number of desired returns in the range of [min(er), max(er)]
        er: A numpy array of expected returns
        cov: A pandas data frame consisting the covariance matrix of the asset returns
    '''

    target_returns = np.linspace(er.min(), er.max(), n_points)

    weights = [minimize_volatility(r, er, cov) for r in target_returns]

    return weights


def MSR(risk_free_return, er, cov):
    '''
    Takes risk_free_return, expected returns of stocks and covariance
    matrix of stocks and returns a weight vector, which can be used to find
    the MSR portfolio

    Inputs:
        risk_free_return: risk free return
        er: expected returns of all the assets
        cov: covariance matrix of all the asset returns
        The risk_free_return and er must be on the same time units
    Output:
        A weight vector

    To find the efficient frontier, we did the following:
    1.  Get a series of returns between the R_Min and R_Max, where R_min ts the
        minimum rate of an asset in the protfolio and R_Max is the maximum
        rate of an asset in the portfolio
    2.  For each return in the returns series:
        a.  Find the optimal weights such that the volatility of the portfolio
            is minimal

    To find the efficient frontier, we do NOT use any series of returns. But we
    will still find the optimal weights by maximizing the sharpe ratio of the
    portfolio.

    Since there is NO maximize() function in scipy.optimize package, we use the
    minimize() function, and try to minimize the negative sharpe ratio
    '''

    # Get the number of assets:
    n = er.shape[0]

    # Initialize weights
    init_guess = np.repeat(1/n, n)

    # Define bounds
    bounds = ((0, 1),) * n

    # Constraints on the sum of weights
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    # Define an inline function to get the negative sharpe ratio
    def neg_sharpe_ratio(weights,
                         risk_free_return,
                         er,
                         cov):
        pr = portfolio_return(weights, er)
        vol = portfolio_volatility(weights, cov)

        return -(pr - risk_free_return)/(vol)

    # Minimize the negative sharpe ratio
    results = minimize(neg_sharpe_ratio,
                       init_guess,
                       args=(risk_free_return, er, cov),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x


def GMV(cov):
    '''
    Find the Global Minimum Variance Portfolio.
    The GMV is only dependent on the covariance matrix.

    Input:
        A covariance matric (Pandas Data Frame)
    Output:
        A weight vector for the minimum variance portfolio
    '''
    # Get the number of assets
    n = cov.shape[0]

    # As we are re-using MSR() function, let us assign
    # some value to the risk_free_return, as it really does not matter
    # what value you assign
    rf = 0

    # The following call will get optimal weights for the postfolio
    # containing assets that have the same returns.
    # This will indirectly minimize the denominator,
    # which is the variance of the portfolio

    return MSR(risk_free_return=rf, er=np.repeat(1, n), cov=cov)


def plot_ef(n_points,
            er,
            cov,
            risk_free_return=0.0,
            show_cml=False,
            show_ew=False,
            show_gmv=False,
            style='.-'):
    '''
    Plots the risk-return curve
    Inputs:
        n_points: Number of target return values to be considered between the
                  minimum return and maximum return (min and max of all asset
                  returns), including the min and max values.
        er: Expected returns of all the assets.
        cov: Covariance matrix of all the asset returns.
        risk_free_rate: Risk Free Rate.
        show_cml: If True, displays the capital Market Line (CML).
        show_ew: If True, displays the Equal Weighted Portfolio.
                 This usually does not lie on the risk-return curve.
                 This portfolio is independent of the assets returns
                 and assets covariance matrix. This means this is free of
                 estimation errors.
         show_gmv: If True, displays the global Minimum Variance portfolio
                 GMV is dependent only on the covariance of the assets returns.
                 It is NOT dependnet on the assets returns.
         sryle: Controls the line style.
    '''
    # Get the optimal weights for the given covariance and expected returns
    # All these weights correspond to a target return.
    # We choose different target returns between the min and max returns of the
    # assets inside the portfolio
    weights = optimal_weights(n_points, er, cov)

    # Generate returns for each set of optimal weights:
    rets = [portfolio_return(w, er) for w in weights]

    # Generate volatility for each set of optimal weights:
    vol = [portfolio_volatility(w, cov) for w in weights]

    # Create a data frame
    ef = pd.DataFrame({'Returns': rets,
                       'Volatility': vol})

    # Plot the risk-return curve
    ax = ef.plot.line(x='Volatility', y='Returns', style=style)

    # Let us plot the equal weighted portfolio
    if show_ew:
        n = er.shape[0]

        # Generate equal weights vector
        w_ew = np.repeat(1/n, n)

        # Get the portfolio return with the above weights
        r_ew = portfolio_return(w_ew, er)

        # Get the portfolio volatility with the above weights
        vol_ew = portfolio_volatility(w_ew, cov)

        # Add EW portfolio point
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)

    # Let us plot the GMV portfolio
    if show_gmv:
        # Get the optimal weights for the minimum variance portfolio
        w_gmv = GMV(cov)

        # Get the portfolio volatility with the above weights
        r_gmv = portfolio_return(w_gmv, er)

        # Get the portfolio volatility with the above weights
        vol_gmv = portfolio_volatility(w_gmv, cov)

        # Add GMV portfolio point
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)

    # Let us plot the CML (Capital Market Line)
    if show_cml:
        ax.set_xlim(left=0)

        # Get the weights of Maximum Sharpe Ratio Protfolio
        w_msr = MSR(risk_free_return, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_volatility(w_msr, cov)

        # Add CML Line
        cml_x = [0, vol_msr]
        cml_y = [risk_free_return, r_msr]

        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed',
                markersize=12, linewidth=2)
        return ax


def main():
    # me_m = read_Portfolios_Formed_on_ME_monthly_EW()
    # print(me_m.head())
    # print(skewness(me_m).head())
    read_edhec_hedgefundindices


# Boiler plate syntax
if __name__ == '__main__':
    main()
