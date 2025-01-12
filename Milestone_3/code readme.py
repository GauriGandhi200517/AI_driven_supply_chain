import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class InventoryAnalyzer:
    def __init__(self):
        # Initialize example data
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample inventory data for demonstration"""
        # Create inventory transactions data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Generate sample inventory data
        self.inventory_data = pd.DataFrame({
            'date': dates.repeat(3),
            'product_id': ['P001', 'P002', 'P003'] * len(dates),
            'quantity': np.random.randint(10, 100, len(dates) * 3),
            'transaction_type': np.random.choice(['in', 'out'], len(dates) * 3),
            'unit_price': np.random.uniform(10, 100, len(dates) * 3).round(2)
        })
        
        # Generate product information
        self.product_info = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003'],
            'product_name': ['Widgets', 'Gadgets', 'Tools'],
            'category': ['Electronics', 'Electronics', 'Hardware'],
            'min_stock': [20, 15, 25],
            'max_stock': [80, 70, 90]
        })
    
    def calculate_current_stock(self):
        """Calculate current stock levels for each product"""
        # Group by product and transaction type to get total ins and outs
        stock_movement = self.inventory_data.groupby(
            ['product_id', 'transaction_type'])['quantity'].sum().unstack(fill_value=0)
        
        # Calculate net stock
        current_stock = pd.DataFrame({
            'current_stock': stock_movement['in'] - stock_movement['out']
        }).reset_index()
        
        # Merge with product info
        return current_stock.merge(self.product_info, on='product_id')
    
    def identify_reorder_needs(self):
        """Identify products that need reordering"""
        current_stock = self.calculate_current_stock()
        
        # Flag products below minimum stock
        reorder_needs = current_stock[
            current_stock['current_stock'] < current_stock['min_stock']
        ].copy()
        
        # Calculate reorder quantity
        reorder_needs['reorder_quantity'] = (
            reorder_needs['max_stock'] - reorder_needs['current_stock']
        )
        
        return reorder_needs[['product_id', 'product_name', 'current_stock', 
                            'min_stock', 'reorder_quantity']]
    
    def analyze_inventory_turnover(self, days=30):
        """Calculate inventory turnover for the specified period"""
        # Get date range for analysis
        end_date = self.inventory_data['date'].max()
        start_date = end_date - timedelta(days=days)
        
        # Filter data for the period
        period_data = self.inventory_data[self.inventory_data['date'] >= start_date]
        
        # Calculate sales (outbound transactions)
        sales = period_data[period_data['transaction_type'] == 'out'].groupby(
            'product_id')['quantity'].sum()
        
        # Calculate average inventory
        inventory_by_date = period_data.pivot_table(
            index='date', 
            columns='product_id', 
            values='quantity',
            aggfunc='sum'
        ).fillna(0)
        
        average_inventory = inventory_by_date.mean()
        
        # Calculate turnover ratio
        turnover = pd.DataFrame({
            'sales_quantity': sales,
            'average_inventory': average_inventory,
            'turnover_ratio': (sales / average_inventory).round(2)
        }).reset_index()
        
        return turnover.merge(self.product_info[['product_id', 'product_name']], 
                            on='product_id')
    
    def generate_inventory_report(self):
        """Generate a comprehensive inventory report"""
        # Current stock status
        current_stock = self.calculate_current_stock()
        
        # Reorder needs
        reorder_needs = self.identify_reorder_needs()
        
        # Turnover analysis
        turnover = self.analyze_inventory_turnover()
        
        # Value of inventory
        inventory_value = self.inventory_data[
            self.inventory_data['transaction_type'] == 'in'
        ].groupby('product_id').agg({
            'quantity': 'sum',
            'unit_price': 'mean'
        })
        inventory_value['total_value'] = (
            inventory_value['quantity'] * inventory_value['unit_price']
        ).round(2)
        
        return {
            'current_stock': current_stock,
            'reorder_needs': reorder_needs,
            'turnover_analysis': turnover,
            'inventory_value': inventory_value
        }

# Example usage
def main():
    # Initialize analyzer
    analyzer = InventoryAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_inventory_report()
    
    # Print report sections
    print("\nCurrent Stock Levels:")
    print(report['current_stock'])
    
    print("\nProducts Needing Reorder:")
    print(report['reorder_needs'])
    
    print("\nInventory Turnover Analysis:")
    print(report['turnover_analysis'])
    
    print("\nInventory Value Analysis:")
    print(report['inventory_value'])

if __name__ == "__main__":
    main()