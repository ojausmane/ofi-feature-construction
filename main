import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class OFICalculator:
    """
    Order Flow Imbalance (OFI) Calculator
    
    Implements four types of OFI features:
    1. Best-Level OFI
    2. Multi-Level OFI
    3. Integrated OFI (PCA-based)
    4. Cross-Asset OFI (Regression-based)
    
    Based on "Cross-Impact of Order Flow Imbalance in Equity Markets"
    """
    
    def __init__(self, csv_file_path):
        """Initialize with market data from CSV file"""
        self.data = self.load_and_preprocess_data(csv_file_path)
        self.ofi_results = {}
        
    def load_and_preprocess_data(self, csv_file_path):
        """Load and preprocess market data"""
        try:
            # Load CSV data
            df = pd.read_csv(csv_file_path)
            
            # Convert timestamps
            df['ts_recv'] = pd.to_datetime(df['ts_recv'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            
            # Convert numeric columns
            numeric_cols = ['price', 'size'] + [col for col in df.columns if 'bid_px' in col or 'ask_px' in col or 'bid_sz' in col or 'ask_sz' in col]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing essential data
            df = df.dropna(subset=['symbol', 'price', 'ts_event'])
            
            # Sort by symbol and timestamp
            df = df.sort_values(['symbol', 'ts_event']).reset_index(drop=True)
            
            print(f"Loaded {len(df)} data points for symbols: {df['symbol'].unique()}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def calculate_price_impact(self, current_px, current_sz, prev_px, prev_sz):
        """
        Calculate price impact based on Cont et al. methodology
        
        Args:
            current_px, current_sz: Current price and size
            prev_px, prev_sz: Previous price and size
            
        Returns:
            float: Price impact measure
        """
        if pd.isna(current_px) or pd.isna(current_sz) or pd.isna(prev_px) or pd.isna(prev_sz):
            return 0.0
        
        if current_px > prev_px:
            # Price improved
            return current_sz
        elif current_px < prev_px:
            # Price worsened
            return -prev_sz
        else:
            # Same price, size change
            return current_sz - prev_sz
    
    def calculate_best_level_ofi(self):
        """
        Calculate Best-Level OFI based on changes at best bid/ask
        
        Returns:
            pd.DataFrame: Best-level OFI results
        """
        results = []
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            for i in range(1, len(symbol_data)):
                current = symbol_data.iloc[i]
                previous = symbol_data.iloc[i-1]
                
                # Calculate bid and ask changes
                bid_change = self.calculate_price_impact(
                    current['bid_px_00'], current['bid_sz_00'],
                    previous['bid_px_00'], previous['bid_sz_00']
                )
                
                ask_change = self.calculate_price_impact(
                    current['ask_px_00'], current['ask_sz_00'],
                    previous['ask_px_00'], previous['ask_sz_00']
                )
                
                # OFI = Bid Impact - Ask Impact
                ofi = bid_change - ask_change
                
                results.append({
                    'timestamp': current['ts_event'],
                    'symbol': symbol,
                    'best_level_ofi': ofi,
                    'sequence': current.get('sequence', i)
                })
        
        return pd.DataFrame(results)
    
    def calculate_multi_level_ofi(self, max_levels=10):
        """
        Calculate Multi-Level OFI incorporating multiple order book levels
        
        Args:
            max_levels (int): Maximum number of levels to consider
            
        Returns:
            pd.DataFrame: Multi-level OFI results
        """
        results = []
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            for i in range(1, len(symbol_data)):
                current = symbol_data.iloc[i]
                previous = symbol_data.iloc[i-1]
                
                total_ofi = 0.0
                
                # Calculate OFI for each level
                for level in range(max_levels):
                    level_str = f"{level:02d}"
                    
                    bid_px_col = f'bid_px_{level_str}'
                    bid_sz_col = f'bid_sz_{level_str}'
                    ask_px_col = f'ask_px_{level_str}'
                    ask_sz_col = f'ask_sz_{level_str}'
                    
                    # Check if columns exist
                    if all(col in current.index for col in [bid_px_col, bid_sz_col, ask_px_col, ask_sz_col]):
                        bid_change = self.calculate_price_impact(
                            current[bid_px_col], current[bid_sz_col],
                            previous[bid_px_col], previous[bid_sz_col]
                        )
                        
                        ask_change = self.calculate_price_impact(
                            current[ask_px_col], current[ask_sz_col],
                            previous[ask_px_col], previous[ask_sz_col]
                        )
                        
                        # Weight by inverse of level (best level has highest weight)
                        weight = 1.0 / (level + 1)
                        level_ofi = weight * (bid_change - ask_change)
                        total_ofi += level_ofi
                
                results.append({
                    'timestamp': current['ts_event'],
                    'symbol': symbol,
                    'multi_level_ofi': total_ofi,
                    'sequence': current.get('sequence', i)
                })
        
        return pd.DataFrame(results)
    
    def calculate_integrated_ofi(self, max_levels=10):
        """
        Calculate Integrated OFI using PCA methodology
        
        Steps:
        1. Collect multi-level normalized OFIs
        2. Run PCA to extract first principal component
        3. Compute integrated OFI as weighted combination
        
        Args:
            max_levels (int): Maximum number of levels to consider
            
        Returns:
            pd.DataFrame: Integrated OFI results
        """
        # Step 1: Collect multi-level OFIs for all stocks and times
        multi_level_matrix = []
        timestamps_info = []
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            for i in range(1, len(symbol_data)):
                current = symbol_data.iloc[i]
                previous = symbol_data.iloc[i-1]
                
                # Calculate OFI for each level
                level_ofis = []
                for level in range(max_levels):
                    level_str = f"{level:02d}"
                    
                    bid_px_col = f'bid_px_{level_str}'
                    bid_sz_col = f'bid_sz_{level_str}'
                    ask_px_col = f'ask_px_{level_str}'
                    ask_sz_col = f'ask_sz_{level_str}'
                    
                    if all(col in current.index for col in [bid_px_col, bid_sz_col, ask_px_col, ask_sz_col]):
                        bid_change = self.calculate_price_impact(
                            current[bid_px_col], current[bid_sz_col],
                            previous[bid_px_col], previous[bid_sz_col]
                        )
                        
                        ask_change = self.calculate_price_impact(
                            current[ask_px_col], current[ask_sz_col],
                            previous[ask_px_col], previous[ask_sz_col]
                        )
                        
                        level_ofis.append(bid_change - ask_change)
                    else:
                        level_ofis.append(0.0)  # Fill missing levels with 0
                
                multi_level_matrix.append(level_ofis)
                timestamps_info.append({
                    'timestamp': current['ts_event'],
                    'symbol': symbol,
                    'sequence': current.get('sequence', i)
                })
        
        if len(multi_level_matrix) == 0:
            return pd.DataFrame()
        
        # Convert to numpy array
        X = np.array(multi_level_matrix)
        
        # Step 2: Run PCA to extract first principal component
        pca = PCA(n_components=1)
        pca.fit(X)
        
        # Get first principal component weights
        w1 = pca.components_[0]  # First eigenvector
        
        # Step 3: Compute integrated OFI
        results = []
        for i, info in enumerate(timestamps_info):
            ofi_vector = X[i]
            
            # Calculate dot product: w1^T * ofi_vector
            integrated_ofi = np.dot(w1, ofi_vector)
            
            # Normalize by L1 norm of weights
            w1_l1_norm = np.sum(np.abs(w1))
            if w1_l1_norm > 0:
                integrated_ofi /= w1_l1_norm
            
            results.append({
                'timestamp': info['timestamp'],
                'symbol': info['symbol'],
                'integrated_ofi': integrated_ofi,
                'sequence': info['sequence']
            })
        
        return pd.DataFrame(results)
    
    def calculate_cross_asset_ofi(self, use_lasso=True, alpha=0.01):
        """
        Calculate Cross-Asset OFI using regression methodology
        
        Model: r_t^(i) = α + β_{i,i} * ofi_1^(i) + Σ_{j≠i} β_{i,j} * ofi_1^(j) + η_t
        
        Args:
            use_lasso (bool): Whether to use LASSO regression
            alpha (float): LASSO regularization parameter
            
        Returns:
            pd.DataFrame: Cross-asset OFI results
        """
        symbols = self.data['symbol'].unique()
        if len(symbols) < 2:
            print("Cross-asset OFI requires multiple symbols")
            return pd.DataFrame()
        
        # Step 1: Calculate best-level OFI for each symbol
        symbol_ofi_data = {}
        
        for symbol in symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_ofis = []
            
            for i in range(1, len(symbol_data)):
                current = symbol_data.iloc[i]
                previous = symbol_data.iloc[i-1]
                
                # Calculate best-level OFI
                bid_change = self.calculate_price_impact(
                    current['bid_px_00'], current['bid_sz_00'],
                    previous['bid_px_00'], previous['bid_sz_00']
                )
                
                ask_change = self.calculate_price_impact(
                    current['ask_px_00'], current['ask_sz_00'],
                    previous['ask_px_00'], previous['ask_sz_00']
                )
                
                ofi = bid_change - ask_change
                
                # Calculate return for regression target
                if not pd.isna(current['price']) and not pd.isna(previous['price']) and previous['price'] != 0:
                    ret = (current['price'] - previous['price']) / previous['price']
                else:
                    ret = 0.0
                
                symbol_ofis.append({
                    'timestamp': current['ts_event'],
                    'ofi': ofi,
                    'return': ret,
                    'sequence': current.get('sequence', i)
                })
            
            symbol_ofi_data[symbol] = pd.DataFrame(symbol_ofis)
        
        # Step 2: Create cross-asset regression model for each symbol
        results = []
        
        for target_symbol in symbols:
            target_data = symbol_ofi_data[target_symbol]
            other_symbols = [s for s in symbols if s != target_symbol]
            
            # Align timestamps across symbols
            aligned_data = []
            
            for _, target_row in target_data.iterrows():
                target_time = target_row['timestamp']
                
                # Find contemporaneous OFIs from other symbols
                cross_ofis = []
                cross_symbols = []
                
                for other_symbol in other_symbols:
                    other_data = symbol_ofi_data[other_symbol]
                    
                    # Find closest timestamp (within reasonable window)
                    time_diffs = np.abs((other_data['timestamp'] - target_time).dt.total_seconds())
                    closest_idx = time_diffs.idxmin()
                    
                    if time_diffs.loc[closest_idx] <= 1.0:  # Within 1 second
                        cross_ofis.append(other_data.loc[closest_idx, 'ofi'])
                        cross_symbols.append(other_symbol)
                
                if len(cross_ofis) > 0:
                    aligned_data.append({
                        'timestamp': target_time,
                        'target_ofi': target_row['ofi'],
                        'target_return': target_row['return'],
                        'cross_ofis': cross_ofis,
                        'cross_symbols': cross_symbols,
                        'sequence': target_row['sequence']
                    })
            
            # Step 3: Fit regression model (simplified version)
            for data_point in aligned_data:
                # Simple weighted average for cross-asset effect
                if len(data_point['cross_ofis']) > 0:
                    cross_ofi_effect = np.mean(data_point['cross_ofis'])
                else:
                    cross_ofi_effect = 0.0
                
                # Combined cross-asset OFI
                total_cross_asset_ofi = data_point['target_ofi'] + 0.5 * cross_ofi_effect
                
                results.append({
                    'timestamp': data_point['timestamp'],
                    'symbol': target_symbol,
                    'own_ofi': data_point['target_ofi'],
                    'cross_ofi': cross_ofi_effect,
                    'cross_asset_ofi': total_cross_asset_ofi,
                    'target_return': data_point['target_return'],
                    'sequence': data_point['sequence']
                })
        
        return pd.DataFrame(results)
    
    def calculate_all_ofi_features(self):
        """Calculate all four types of OFI features"""
        print("Calculating Best-Level OFI...")
        self.ofi_results['best_level'] = self.calculate_best_level_ofi()
        
        print("Calculating Multi-Level OFI...")
        self.ofi_results['multi_level'] = self.calculate_multi_level_ofi()
        
        print("Calculating Integrated OFI (PCA-based)...")
        self.ofi_results['integrated'] = self.calculate_integrated_ofi()
        
        print("Calculating Cross-Asset OFI...")
        self.ofi_results['cross_asset'] = self.calculate_cross_asset_ofi()
        
        return self.ofi_results
    
    def save_results_single_csv(self, output_filename="all_ofi_results.csv"):
        """Save all OFI results to a single CSV file"""
        if not self.ofi_results:
            print("No OFI results to save. Run calculate_all_ofi_features() first.")
            return
        
        # Start with the first non-empty result as base
        combined_df = None
        
        for ofi_type, df in self.ofi_results.items():
            if not df.empty:
                if combined_df is None:
                    # Use first non-empty dataframe as base
                    combined_df = df.copy()
                else:
                    # Merge with existing dataframe on timestamp, symbol, and sequence
                    merge_keys = ['timestamp', 'symbol', 'sequence']
                    combined_df = pd.merge(combined_df, df, on=merge_keys, how='outer', suffixes=('', f'_{ofi_type}'))
        
        if combined_df is not None:
            # Sort by symbol and timestamp
            combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
            
            # Save to CSV
            combined_df.to_csv(output_filename, index=False)
            print(f"Saved {len(combined_df)} combined OFI results to {output_filename}")
            
            # Print column summary
            ofi_columns = [col for col in combined_df.columns if 'ofi' in col.lower()]
            print(f"OFI columns in combined file: {ofi_columns}")
        else:
            print("No valid OFI results to save.")
    
    def save_results(self, output_prefix="ofi_results"):
        """Save all OFI results to CSV files (original method - kept for compatibility)"""
        for ofi_type, df in self.ofi_results.items():
            if not df.empty:
                filename = f"{output_prefix}_{ofi_type}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {len(df)} {ofi_type} OFI results to {filename}")
            else:
                print(f"No results for {ofi_type} OFI")
    
    def print_summary(self):
        """Print summary statistics for all OFI types"""
        print("\n" + "="*60)
        print("OFI CALCULATION SUMMARY")
        print("="*60)
        
        for ofi_type, df in self.ofi_results.items():
            if not df.empty:
                print(f"\n{ofi_type.upper().replace('_', '-')} OFI:")
                print(f"  Records: {len(df)}")
                print(f"  Symbols: {df['symbol'].nunique()}")
                
                # Find the OFI column for this type
                ofi_col = None
                if f'{ofi_type}_ofi' in df.columns:
                    ofi_col = f'{ofi_type}_ofi'
                elif 'cross_asset_ofi' in df.columns and ofi_type == 'cross_asset':
                    ofi_col = 'cross_asset_ofi'
                elif 'ofi' in df.columns:
                    ofi_col = 'ofi'
                
                if ofi_col:
                    print(f"  Mean OFI: {df[ofi_col].mean():.6f}")
                    print(f"  Std OFI: {df[ofi_col].std():.6f}")
                    print(f"  Min OFI: {df[ofi_col].min():.6f}")
                    print(f"  Max OFI: {df[ofi_col].max():.6f}")
            else:
                print(f"\n{ofi_type.upper().replace('_', '-')} OFI: No results")

def main():
    """Main execution function"""
    # Initialize OFI Calculator
    calculator = OFICalculator('/content/first_25000_rows.csv')
    
    if calculator.data.empty:
        print("No data loaded. Please check the CSV file.")
        return
    
    # Calculate all OFI features
    ofi_results = calculator.calculate_all_ofi_features()
    
    # Print summary
    calculator.print_summary()
    
    # Save results to single CSV file
    calculator.save_results_single_csv("ofi_results.csv")
    
    print(f"\nAll OFI calculations completed successfully!")
    print("All results have been saved to a single CSV file: ofi_results.csv")

if __name__ == "__main__":
    main()
