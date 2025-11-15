import numpy as np
import pandas as pd
from typing import Optional


def stationary_bootstrap(traindf: pd.DataFrame, sample_size: int, 
                        avg_block_length: float = 10.0, 
                        random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate a bootstrap sample using the stationary bootstrap method.
    
    The stationary bootstrap preserves the time series structure by using 
    random block lengths drawn from a geometric distribution.
    
    Parameters:
    -----------
    traindf : pd.DataFrame
        The training dataframe to bootstrap from. Should have time series data
        with dates as index and assets as columns.
    sample_size : int
        The desired number of observations in the bootstrap sample.
    avg_block_length : float, default=10.0
        The average block length for the geometric distribution.
        Higher values preserve more temporal dependence.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        A bootstrap sample with the same column structure as traindf
        but with sample_size rows.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_obs, n_assets = traindf.shape
    
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if avg_block_length <= 0:
        raise ValueError("avg_block_length must be positive")
    if n_obs == 0:
        raise ValueError("traindf cannot be empty")
    
    # Probability parameter for geometric distribution
    # E[X] = 1/p for geometric distribution, so p = 1/avg_block_length
    p = 1.0 / avg_block_length
    
    # Initialize the bootstrap sample
    bootstrap_data = []
    current_length = 0
    
    while current_length < sample_size:
        # Draw a random starting point
        start_idx = np.random.randint(0, n_obs)
        
        # Draw block length from geometric distribution
        # np.random.geometric returns values starting from 1
        block_length = np.random.geometric(p)
        
        # Extract the block (with wraparound if necessary)
        block_indices = []
        for i in range(block_length):
            idx = (start_idx + i) % n_obs
            block_indices.append(idx)
        
        # Add the block to our bootstrap sample
        block_data = traindf.iloc[block_indices].values
        bootstrap_data.append(block_data)
        current_length += block_length
    
    # Concatenate all blocks and trim to desired sample size
    bootstrap_array = np.vstack(bootstrap_data)[:sample_size]
    
    # Create new DataFrame with appropriate index
    # Generate new date index (could be sequential or based on original dates)
    if isinstance(traindf.index, pd.DatetimeIndex):
        # Create a new date range starting from the first date
        start_date = traindf.index[0]
        freq = pd.infer_freq(traindf.index) or 'D'  # Default to daily if can't infer
        new_index = pd.date_range(start=start_date, periods=sample_size, freq=freq)
    else:
        # Create a simple integer index
        new_index = range(sample_size)
    
    bootstrap_df = pd.DataFrame(
        bootstrap_array,
        index=new_index,
        columns=traindf.columns
    )
    
    return bootstrap_df


def generate_bootstrap_samples(traindf: pd.DataFrame, 
                             n_samples: int,
                             sample_size: int,
                             avg_block_length: float = 10.0,
                             random_state: Optional[int] = None) -> list:
    """
    Generate multiple bootstrap samples.
    
    Parameters:
    -----------
    traindf : pd.DataFrame
        The training dataframe to bootstrap from.
    n_samples : int
        Number of bootstrap samples to generate.
    sample_size : int
        The desired number of observations in each bootstrap sample.
    avg_block_length : float, default=10.0
        The average block length for the geometric distribution.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns:
    --------
    list
        List of bootstrap sample DataFrames.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bootstrap_samples = []
    for i in range(n_samples):
        # Use different random states for each sample if seed is provided
        sample_seed = None if random_state is None else random_state + i
        sample = stationary_bootstrap(
            traindf, 
            sample_size, 
            avg_block_length, 
            sample_seed
        )
        bootstrap_samples.append(sample)
    
    return bootstrap_samples
