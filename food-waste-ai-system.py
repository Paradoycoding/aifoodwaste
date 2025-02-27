# AI Food Waste Management System
# Main application architecture

import os
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class InventoryItem:
    def __init__(self, item_id, name, category, unit, shelf_life_days, current_quantity, 
                 reorder_point, supplier_id, cost_per_unit):
        self.item_id = item_id
        self.name = name
        self.category = category
        self.unit = unit
        self.shelf_life_days = shelf_life_days
        self.current_quantity = current_quantity
        self.reorder_point = reorder_point
        self.supplier_id = supplier_id
        self.cost_per_unit = cost_per_unit
        self.expiry_dates = {}  # Dictionary mapping batch_id to expiry date
        
    def add_stock(self, quantity, batch_id, expiry_date):
        self.current_quantity += quantity
        self.expiry_dates[batch_id] = expiry_date
        print(f"Added {quantity} {self.unit} of {self.name}, Batch ID: {batch_id}, Expiry: {expiry_date}")
        
    def remove_stock(self, quantity):
        if self.current_quantity >= quantity:
            self.current_quantity -= quantity
            # Remove from oldest batch first (FIFO)
            qty_to_remove = quantity
            batches_to_remove = []
            
            sorted_batches = sorted(self.expiry_dates.items(), key=lambda x: x[1])
            for batch_id, expiry in sorted_batches:
                if qty_to_remove <= 0:
                    break
                # Assuming each batch has equal quantity (can be enhanced)
                batch_qty = qty_to_remove
                batches_to_remove.append(batch_id)
                qty_to_remove = 0
            
            for batch_id in batches_to_remove:
                del self.expiry_dates[batch_id]
                
            return True
        else:
            print(f"Insufficient stock of {self.name}")
            return False
    
    def check_expiring_items(self, days_threshold=3):
        """Check items expiring within the threshold days"""
        today = datetime.datetime.now().date()
        expiring_soon = {}
        
        for batch_id, expiry in self.expiry_dates.items():
            if isinstance(expiry, str):
                expiry = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
            days_remaining = (expiry - today).days
            if 0 <= days_remaining <= days_threshold:
                expiring_soon[batch_id] = days_remaining
        
        return expiring_soon


class InventoryManager:
    def __init__(self):
        self.inventory = {}  # Dictionary of InventoryItem objects
        self.waste_log = []
        self.usage_history = []
        self.purchase_history = []
        
    def add_item(self, item):
        self.inventory[item.item_id] = item
        
    def record_waste(self, item_id, quantity, reason, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        waste_entry = {
            'item_id': item_id,
            'item_name': self.inventory[item_id].name if item_id in self.inventory else "Unknown",
            'quantity': quantity,
            'unit': self.inventory[item_id].unit if item_id in self.inventory else "unit",
            'reason': reason,
            'timestamp': timestamp,
            'cost': self.inventory[item_id].cost_per_unit * quantity if item_id in self.inventory else 0
        }
        
        self.waste_log.append(waste_entry)
        print(f"Recorded waste: {quantity} {waste_entry['unit']} of {waste_entry['item_name']}")
        
    def get_expiring_items(self, days_threshold=3):
        expiring_items = {}
        for item_id, item in self.inventory.items():
            expiring_batches = item.check_expiring_items(days_threshold)
            if expiring_batches:
                expiring_items[item_id] = {
                    'name': item.name,
                    'batches': expiring_batches
                }
        return expiring_items
    
    def generate_waste_report(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.datetime.now()
            
        filtered_waste = [entry for entry in self.waste_log 
                         if start_date <= entry['timestamp'] <= end_date]
        
        # Aggregate by item
        waste_by_item = {}
        for entry in filtered_waste:
            item_id = entry['item_id']
            if item_id not in waste_by_item:
                waste_by_item[item_id] = {
                    'name': entry['item_name'],
                    'total_quantity': 0,
                    'total_cost': 0,
                    'reasons': {}
                }
            
            waste_by_item[item_id]['total_quantity'] += entry['quantity']
            waste_by_item[item_id]['total_cost'] += entry['cost']
            
            # Track reasons
            reason = entry['reason']
            if reason not in waste_by_item[item_id]['reasons']:
                waste_by_item[item_id]['reasons'][reason] = 0
            waste_by_item[item_id]['reasons'][reason] += entry['quantity']
        
        return waste_by_item
    
    def recommend_actions_for_expiring_items(self):
        expiring_items = self.get_expiring_items(5)  # 5-day threshold
        recommendations = []
        
        for item_id, details in expiring_items.items():
            item = self.inventory[item_id]
            
            # Get total expiring quantity (estimate)
            expiring_qty = len(details['batches'])  # This is simplified
            
            if item.category == 'Produce':
                if expiring_qty > 3:  # Arbitrary threshold
                    recommendations.append({
                        'item_id': item_id,
                        'name': item.name,
                        'action': 'Create special dish',
                        'priority': 'High' if len(details['batches']) > 0 else 'Medium',
                        'details': f"Use {item.name} in daily special to reduce waste"
                    })
                else:
                    recommendations.append({
                        'item_id': item_id,
                        'name': item.name,
                        'action': 'Prep for staff meal',
                        'priority': 'Medium',
                        'details': f"Incorporate {item.name} into staff meal"
                    })
            elif item.category == 'Dairy':
                recommendations.append({
                    'item_id': item_id,
                    'name': item.name,
                    'action': 'Check for donation eligibility',
                    'priority': 'High',
                    'details': f"Verify if {item.name} can be donated before expiry"
                })
            elif item.category == 'Meat':
                recommendations.append({
                    'item_id': item_id,
                    'name': item.name,
                    'action': 'Immediate menu special',
                    'priority': 'High',
                    'details': f"Create protein special using {item.name}"
                })
            else:
                recommendations.append({
                    'item_id': item_id,
                    'name': item.name,
                    'action': 'Review usage',
                    'priority': 'Medium',
                    'details': f"Assess why {item.name} is not being used as expected"
                })
        
        return recommendations


class Recipe:
    def __init__(self, recipe_id, name, category, ingredients, preparation_steps, 
                 cooking_time, portion_size, selling_price):
        self.recipe_id = recipe_id
        self.name = name
        self.category = category
        self.ingredients = ingredients  # List of dicts with item_id, quantity
        self.preparation_steps = preparation_steps
        self.cooking_time = cooking_time
        self.portion_size = portion_size
        self.selling_price = selling_price
        self.popularity_score = 0
        self.waste_percentage = 0
        
    def calculate_cost(self, inventory_manager):
        total_cost = 0
        for ingredient in self.ingredients:
            item_id = ingredient['item_id']
            quantity = ingredient['quantity']
            if item_id in inventory_manager.inventory:
                item = inventory_manager.inventory[item_id]
                total_cost += item.cost_per_unit * quantity
            else:
                print(f"Warning: Item {item_id} not found in inventory")
        return total_cost
    
    def calculate_profit_margin(self, inventory_manager):
        cost = self.calculate_cost(inventory_manager)
        if cost > 0:
            return (self.selling_price - cost) / self.selling_price * 100
        return 0


class MenuPlanner:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        self.recipes = {}
        self.menu_items = []  # Active menu
        self.sales_history = []
        self.ai_model = None
    
    def add_recipe(self, recipe):
        self.recipes[recipe.recipe_id] = recipe
    
    def add_to_menu(self, recipe_id):
        if recipe_id in self.recipes:
            self.menu_items.append(recipe_id)
            print(f"Added {self.recipes[recipe_id].name} to the menu")
        else:
            print(f"Recipe {recipe_id} not found")
    
    def remove_from_menu(self, recipe_id):
        if recipe_id in self.menu_items:
            self.menu_items.remove(recipe_id)
            print(f"Removed {self.recipes[recipe_id].name} from the menu")
        else:
            print(f"Recipe {recipe_id} not on the menu")
    
    def record_sale(self, recipe_id, quantity, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        if recipe_id in self.recipes:
            recipe = self.recipes[recipe_id]
            
            # Deduct ingredients from inventory
            all_ingredients_available = True
            for ingredient in recipe.ingredients:
                item_id = ingredient['item_id']
                required_qty = ingredient['quantity'] * quantity
                
                if item_id in self.inventory_manager.inventory:
                    if not self.inventory_manager.inventory[item_id].remove_stock(required_qty):
                        all_ingredients_available = False
                        print(f"Cannot fulfill order: insufficient {self.inventory_manager.inventory[item_id].name}")
                else:
                    all_ingredients_available = False
                    print(f"Item {item_id} not found in inventory")
            
            if all_ingredients_available:
                sale_entry = {
                    'recipe_id': recipe_id,
                    'recipe_name': recipe.name,
                    'quantity': quantity,
                    'timestamp': timestamp,
                    'revenue': recipe.selling_price * quantity,
                    'cost': recipe.calculate_cost(self.inventory_manager) * quantity
                }
                
                self.sales_history.append(sale_entry)
                print(f"Recorded sale: {quantity} of {recipe.name}")
                return True
            else:
                print("Sale not recorded due to inventory issues")
                return False
        else:
            print(f"Recipe {recipe_id} not found")
            return False
    
    def train_demand_prediction_model(self):
        """Train an AI model to predict demand based on historical sales"""
        if len(self.sales_history) < 30:  # Need minimum data
            print("Insufficient sales history for AI training")
            return False
        
        # Prepare training data
        data = []
        for sale in self.sales_history:
            # Extract features: day of week, month, recipe category, etc.
            day_of_week = sale['timestamp'].weekday()
            month = sale['timestamp'].month
            recipe = self.recipes[sale['recipe_id']]
            category = recipe.category
            # Convert category to numerical (simplified)
            category_code = hash(category) % 10
            
            # More features can be added
            features = [day_of_week, month, category_code, recipe.popularity_score]
            target = sale['quantity']
            
            data.append(features + [target])
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['day_of_week', 'month', 'category', 'popularity', 'quantity'])
        
        # Split data
        X = df.drop('quantity', axis=1)
        y = df['quantity']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        score = model.score(X_test_scaled, y_test)
        print(f"Model R² score: {score:.4f}")
        
        # Save model and scaler
        self.ai_model = {
            'model': model,
            'scaler': scaler
        }
        
        # Save to disk
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/demand_prediction_model.pkl')
        joblib.dump(scaler, 'models/feature_scaler.pkl')
        
        return True
    
    def predict_demand(self, recipe_id, future_date):
        """Predict demand for a recipe on a specific future date"""
        if self.ai_model is None:
            print("AI model not trained yet")
            return None
            
        if recipe_id not in self.recipes:
            print(f"Recipe {recipe_id} not found")
            return None
            
        recipe = self.recipes[recipe_id]
        
        # Extract features
        day_of_week = future_date.weekday()
        month = future_date.month
        category_code = hash(recipe.category) % 10
        
        # Create feature array
        features = np.array([[day_of_week, month, category_code, recipe.popularity_score]])
        
        # Scale features
        features_scaled = self.ai_model['scaler'].transform(features)
        
        # Predict
        predicted_demand = self.ai_model['model'].predict(features_scaled)[0]
        
        return max(0, round(predicted_demand))
    
    def generate_menu_recommendations(self):
        """Generate menu recommendations based on inventory, waste data, and sales"""
        recommendations = []
        
        # Check expiring items
        expiring_items = self.inventory_manager.get_expiring_items(5)
        
        # Find recipes that use expiring items
        recipes_with_expiring = []
        for item_id in expiring_items:
            for recipe_id, recipe in self.recipes.items():
                for ingredient in recipe.ingredients:
                    if ingredient['item_id'] == item_id:
                        recipes_with_expiring.append({
                            'recipe_id': recipe_id,
                            'recipe_name': recipe.name,
                            'item_id': item_id,
                            'item_name': self.inventory_manager.inventory[item_id].name,
                            'days_remaining': min(expiring_items[item_id]['batches'].values())
                        })
        
        # Sort by days remaining (ascending)
        recipes_with_expiring.sort(key=lambda x: x['days_remaining'])
        
        # Add to recommendations
        for recipe in recipes_with_expiring:
            if recipe['recipe_id'] not in self.menu_items:
                recommendations.append({
                    'action': 'Add to menu',
                    'recipe_id': recipe['recipe_id'],
                    'recipe_name': recipe['recipe_name'],
                    'reason': f"Uses {recipe['item_name']} which expires in {recipe['days_remaining']} days",
                    'priority': 'High' if recipe['days_remaining'] <= 2 else 'Medium'
                })
        
        # Analyze sales history for underperforming menu items
        if len(self.sales_history) > 0:
            # Group by recipe
            sales_by_recipe = {}
            for sale in self.sales_history:
                recipe_id = sale['recipe_id']
                if recipe_id not in sales_by_recipe:
                    sales_by_recipe[recipe_id] = {
                        'total_quantity': 0,
                        'total_revenue': 0
                    }
                sales_by_recipe[recipe_id]['total_quantity'] += sale['quantity']
                sales_by_recipe[recipe_id]['total_revenue'] += sale['revenue']
            
            # Find underperforming items (simplified)
            for recipe_id in self.menu_items:
                if recipe_id in sales_by_recipe:
                    if sales_by_recipe[recipe_id]['total_quantity'] < 10:  # Arbitrary threshold
                        recommendations.append({
                            'action': 'Consider removing',
                            'recipe_id': recipe_id,
                            'recipe_name': self.recipes[recipe_id].name,
                            'reason': f"Low sales volume ({sales_by_recipe[recipe_id]['total_quantity']} units)",
                            'priority': 'Medium'
                        })
                else:
                    # No sales at all
                    recommendations.append({
                        'action': 'Remove from menu',
                        'recipe_id': recipe_id,
                        'recipe_name': self.recipes[recipe_id].name,
                        'reason': "No sales recorded",
                        'priority': 'High'
                    })
        
        return recommendations


class WasteAnalyzer:
    def __init__(self, inventory_manager):
        self.inventory_manager = inventory_manager
        
    def identify_waste_patterns(self):
        """Identify patterns in waste data"""
        waste_log = self.inventory_manager.waste_log
        
        if len(waste_log) < 10:  # Need minimum data
            return {"message": "Insufficient waste data for analysis"}
        
        # Aggregate by reason
        waste_by_reason = {}
        for entry in waste_log:
            reason = entry['reason']
            if reason not in waste_by_reason:
                waste_by_reason[reason] = {
                    'count': 0,
                    'total_quantity': 0,
                    'total_cost': 0,
                    'items': {}
                }
            
            waste_by_reason[reason]['count'] += 1
            waste_by_reason[reason]['total_quantity'] += entry['quantity']
            waste_by_reason[reason]['total_cost'] += entry['cost']
            
            item_id = entry['item_id']
            if item_id not in waste_by_reason[reason]['items']:
                waste_by_reason[reason]['items'][item_id] = {
                    'name': entry['item_name'],
                    'quantity': 0,
                    'cost': 0
                }
            
            waste_by_reason[reason]['items'][item_id]['quantity'] += entry['quantity']
            waste_by_reason[reason]['items'][item_id]['cost'] += entry['cost']
        
        # Sort reasons by cost (descending)
        sorted_reasons = sorted(waste_by_reason.items(), 
                               key=lambda x: x[1]['total_cost'], 
                               reverse=True)
        
        # Prepare result
        result = {
            'top_reasons': [],
            'total_waste_cost': sum(item['total_cost'] for item in waste_by_reason.values()),
            'recommendation': ""
        }
        
        # Add top 3 reasons
        for reason, data in sorted_reasons[:3]:
            result['top_reasons'].append({
                'reason': reason,
                'count': data['count'],
                'total_quantity': data['total_quantity'],
                'total_cost': data['total_cost'],
                'percentage_of_total': data['total_cost'] / result['total_waste_cost'] * 100,
                'most_wasted_items': sorted(data['items'].items(), 
                                          key=lambda x: x[1]['cost'], 
                                          reverse=True)[:3]
            })
        
        # Generate recommendation
        if len(result['top_reasons']) > 0:
            top_reason = result['top_reasons'][0]
            if top_reason['reason'] == 'Expired':
                result['recommendation'] = "Improve inventory management and purchasing forecasts to reduce expiry waste."
            elif top_reason['reason'] == 'Overproduction':
                result['recommendation'] = "Adjust batch sizes and implement just-in-time cooking strategies."
            elif top_reason['reason'] == 'Trim waste':
                result['recommendation'] = "Review preparation techniques and train staff on efficient cutting methods."
            else:
                result['recommendation'] = f"Investigate and address the primary waste cause: {top_reason['reason']}"
        
        return result
    
    def suggest_waste_reduction_strategies(self):
        """Suggest strategies to reduce food waste"""
        waste_patterns = self.identify_waste_patterns()
        
        if 'message' in waste_patterns:
            return {"message": waste_patterns['message']}
        
        strategies = []
        
        # General strategies
        strategies.append({
            'category': 'General',
            'strategies': [
                "Implement daily waste tracking by category",
                "Regular staff training on waste reduction",
                "Set waste reduction targets and incentives"
            ]
        })
        
        # Specific strategies based on waste patterns
        if len(waste_patterns['top_reasons']) > 0:
            for reason_data in waste_patterns['top_reasons']:
                reason = reason_data['reason']
                
                if reason == 'Expired':
                    strategies.append({
                        'category': 'Inventory Management',
                        'strategies': [
                            "Implement FIFO (First In, First Out) labeling system",
                            "Reduce order quantities for frequently expiring items",
                            "Review storage conditions and temperatures",
                            "Create 'use first' section in storage areas"
                        ]
                    })
                elif reason == 'Overproduction':
                    strategies.append({
                        'category': 'Production Planning',
                        'strategies': [
                            "Implement batch cooking during service",
                            "Use historical sales data to forecast demand",
                            "Standardize recipes and portion sizes",
                            "Develop creative uses for leftover ingredients"
                        ]
                    })
                elif reason == 'Trim waste':
                    strategies.append({
                        'category': 'Preparation Techniques',
                        'strategies': [
                            "Train staff on efficient cutting techniques",
                            "Use trim waste for stocks and sauces",
                            "Audit preparation processes to identify improvement areas",
                            "Invest in appropriate tools for efficient preparation"
                        ]
                    })
                elif reason == 'Plate waste':
                    strategies.append({
                        'category': 'Menu Engineering',
                        'strategies': [
                            "Review portion sizes",
                            "Offer half-portion options",
                            "Solicit customer feedback on portion satisfaction",
                            "Analyze which dishes have highest plate waste"
                        ]
                    })
        
        # Add recommendations for most wasted items
        most_wasted_items = []
        for reason_data in waste_patterns['top_reasons']:
            for item_id, item_data in reason_data['most_wasted_items']:
                most_wasted_items.append({
                    'id': item_id,
                    'name': item_data['name'],
                    'cost': item_data['cost'],
                    'reason': reason_data['reason']
                })
        
        most_wasted_items.sort(key=lambda x: x['cost'], reverse=True)
        
        if len(most_wasted_items) > 0:
            item_strategies = []
            for item in most_wasted_items[:3]:
                item_strategies.append(f"For {item['name']} ({item['reason']}): " + 
                                     self._get_item_specific_strategy(item['name'], item['reason']))
            
            strategies.append({
                'category': 'Item-Specific Actions',
                'strategies': item_strategies
            })
        
        return {
            'total_waste_cost': waste_patterns['total_waste_cost'],
            'strategies': strategies
        }
    
    def _get_item_specific_strategy(self, item_name, reason):
        """Generate item-specific waste reduction strategy"""
        # This would use more complex logic in a real system
        if reason == 'Expired':
            return f"Reduce order quantity and increase usage in daily specials"
        elif reason == 'Overproduction':
            return f"Implement just-in-time preparation and repurpose excess in new dishes"
        elif reason == 'Trim waste':
            return f"Review preparation method and utilize trim in stocks or purees"
        else:
            return f"Analyze usage patterns and adjust ordering/preparation accordingly"


class AIFoodWasteSystem:
    def __init__(self):
        self.inventory_manager = InventoryManager()
        self.menu_planner = MenuPlanner(self.inventory_manager)
        self.waste_analyzer = WasteAnalyzer(self.inventory_manager)
        
    def initialize_demo_data(self):
        """Initialize with demo data for testing"""
        # Add inventory items
        items = [
            InventoryItem("I001", "Tomatoes", "Produce", "kg", 7, 10, 5, "S001", 2.50),
            InventoryItem("I002", "Chicken Breast", "Meat", "kg", 4, 15, 7, "S002", 8.75),
            InventoryItem("I003", "Pasta", "Dry Goods", "kg", 365, 20, 10, "S003", 1.25),
            InventoryItem("I004", "Heavy Cream", "Dairy", "liter", 14, 5, 3, "S001", 3.50),
            InventoryItem("I005", "Lettuce", "Produce", "head", 5, 8, 4, "S001", 1.75),
            InventoryItem("I006", "Beef Mince", "Meat", "kg", 3, 12, 6, "S002", 9.50),
            InventoryItem("I007", "Rice", "Dry Goods", "kg", 365, 25, 10, "S003", 1.80),
            InventoryItem("I008", "Cheese", "Dairy", "kg", 21, 7, 3, "S001", 12.00)
        ]
        
        for item in items:
            self.inventory_manager.add_item(item)
            # Add some stock with expiry dates
            today = datetime.datetime.now().date()
            item.add_stock(item.current_quantity, "B001", today + datetime.timedelta(days=item.shelf_life_days))
        
        # Add recipes
        recipes = [
            Recipe("R001", "Spaghetti Bolognese", "Main", 
                  [{"item_id": "I003", "quantity": 0.2}, 
                   {"item_id": "I006", "quantity": 0.25},
                   {"item_id": "I001", "quantity": 0.1}], 
                  ["Cook pasta", "Make sauce", "Combine"], 
                  25, 1, 15.99),
            
            Recipe("R002", "Caesar Salad", "Starter", 
                  [{"item_id": "I005", "quantity": 0.5}, 
                   {"item_id": "I002", "quantity": 0.15},
                   {"item_id": "I008", "quantity": 0.05}], 
                  ["Prepare lettuce", "Cook chicken", "Assemble"], 
                  15, 1, 12.50),
            
            Recipe("R003", "Chicken Alfredo", "Main", 
                  [{"item_id": "I003", "quantity": 0.2}, 
                   {"item_id": "I002", "quantity": 0.2},
                   {"item_id": "I004", "quantity": 0.1},
                   {"item_id": "I008", "quantity": 0.07}], 
                  ["Cook pasta", "Prepare sauce", "Cook chicken", "Combine"], 
                  30, 1, 17.99)
        ]
        
        for recipe in recipes:
            self.menu_planner.add_recipe(recipe)
            self.menu_planner.add_to_menu(recipe.recipe_id)
        
        # Add some waste records
        waste_reasons = ["Expired", "Overproduction", "Trim waste", "Spoiled", "Spilled"]
        
        # Create waste entries over the past 30 days
        for i in range(30):
            day = datetime.datetime.now() - datetime.timedelta(days=i)
            
            # Random waste entries
            for _ in range(3):
                item_id = f"I00{np.random.randint(1, 9)}"
                if item_id in self.inventory_manager.inventory:
                    quantity = np.random.uniform(0.1, 2.0)
                    reason = np.random.choice(waste_reasons)
                    self.inventory_manager.record_waste(item_id, quantity, reason, day)
        
        # Add some sales history
        for i in range(60):
            day = datetime.datetime.now() - datetime.timedelta(days=i)
            
            # Random sales for each day
            for recipe_id in ["R001", "R002", "R003"]:
                # More sales on weekends
                quantity_factor = 2 if day.weekday() >= 5 else 1  
                quantity = np.random.randint(3, 15) * quantity_factor
                
                # Record without actually removing from inventory (for demo)
                self.menu_planner.sales_history.append({
                    'recipe_id': recipe_id,
                    'recipe_name': self.menu_planner.recipes[recipe_id].name,
                    'quantity': quantity,
                    'timestamp': day,
                    'revenue': self.menu_planner.recipes[recipe_id].selling_price * quantity,
                    'cost': self.menu_planner.recipes[recipe_id].calculate_cost(self.inventory_manager) * quantity
                })
        
        # Train the AI model with this demo data
        self.menu_planner.train_demand_prediction_model()
        
    def suggest_menu_for_sustainability(self, target_date=None):
        """Generate a sustainable menu suggestion to minimize waste"""
        if target_date is None:
            target_date = datetime.datetime.now().date() + datetime.timedelta(days=1)
            
        menu_suggestion = {
            'target_date': target_date,
            'items': [],
            'waste_reduction_potential': 0.0,
            'sustainability_score': 0.0
        }
        
        # Get expiring items
        expiring_items = self.inventory_manager.get_expiring_items(7)  # 7-day window
        
        # Calculate waste reduction potential (in cost)
        potential_waste_cost = 0
        for item_id, details in expiring_items.items():
            item = self.inventory_manager.inventory[item_id]
            # Estimate quantity based on batches (simplified)
            qty = len(details['batches'])
            potential_waste_cost += qty * item.cost_per_unit
        
        menu_suggestion['waste_reduction_potential'] = potential_waste_cost
        
        # Find recipes using expiring items
        recipe_scores = {}
        for recipe_id, recipe in self.menu_planner.recipes.items():
            recipe_scores[recipe_id] = {
                'expiring_usage': 0,
                'popularity': recipe.popularity_score,
                'profit_margin': recipe.calculate_profit_margin(self.inventory_manager),
                'name': recipe.name,
                'category': recipe.category
            }
            
            # Check ingredients
            for ingredient in recipe.ingredients:
                item_id = ingredient['item_id']
                if item_id in expiring_items:
                    # More points for items expiring sooner
                    min_days = min(expiring_items[item_id]['batches'].values())
                    expiry_urgency = max(0, 8 - min_days) # 7 days -> score of 1, 1 day -> score of 7
                    recipe_scores[recipe_id]['expiring_usage'] += expiry_urgency
        
        # Calculate overall score (weighted sum)
        for recipe_id, scores in recipe_scores.items():
            overall_score = (
                scores['expiring_usage'] * 0.6 +  # Prioritize using expiring items
                scores['popularity'] * 0.25 +      # Consider popularity
                scores['profit_margin'] * 0.15     # Consider profit margin
            )
            recipe_scores[recipe_id]['overall_score'] = overall_score
        
        # Sort recipes by overall score (descending)
        sorted_recipes = sorted(recipe_scores.items(), 
                               key=lambda x: x[1]['overall_score'], 
                               reverse=True)
        
        # Create balanced menu (starters, mains, desserts)
        categories = {'Starter': 0, 'Main': 0, 'Dessert': 0}
        max_per_category = {'Starter': 2, 'Main': 3, 'Dessert': 2}
        
        for recipe_id, scores in sorted_recipes:
            category = scores['category']
            if categories.get(category, 0) < max_per_category.get(category, 0):
                menu_suggestion['items'].append({
                    'recipe_id': recipe_id,
                    'name': scores['name'],
                    'category': category,
                    'expiring_usage': scores['expiring_usage'],
                    'score': scores['overall_score'],
                    'predicted_demand': self.menu_planner.predict_demand(recipe_id, target_date)
                })
                categories[category] = categories.get(category, 0) + 1
        
        # Calculate sustainability score (0-100)
        if len(menu_suggestion['items']) > 0:
            avg_expiring_usage = sum(item['expiring_usage'] for item in menu_suggestion['items']) / len(menu_suggestion['items'])
            menu_suggestion['sustainability_score'] = min(100, avg_expiring_usage * 15)
        
        return menu_suggestion
    
    def generate_inventory_order_recommendations(self):
        """Generate inventory order recommendations based on sales forecast"""
        recommendations = {
            'items_to_order': [],
            'items_to_reduce': [],
            'total_estimated_cost': 0
        }
        
        # Check inventory levels
        for item_id, item in self.inventory_manager.inventory.items():
            if item.current_quantity <= item.reorder_point:
                # Calculate order quantity
                order_quantity = (item.reorder_point * 2) - item.current_quantity
                
                recommendations['items_to_order'].append({
                    'item_id': item_id,
                    'name': item.name,
                    'current_quantity': item.current_quantity,
                    'recommended_order': order_quantity,
                    'unit': item.unit,
                    'estimated_cost': order_quantity * item.cost_per_unit,
                    'reason': 'Below reorder point'
                })
                
                recommendations['total_estimated_cost'] += order_quantity * item.cost_per_unit
            
            # Check for excess inventory
            expiry_data = item.check_expiring_items(7)
            if len(expiry_data) > 0 and item.current_quantity > item.reorder_point * 1.5:
                recommendations['items_to_reduce'].append({
                    'item_id': item_id,
                    'name': item.name,
                    'current_quantity': item.current_quantity,
                    'recommended_adjustment': 'Reduce next order by 25%',
                    'expiring_soon': len(expiry_data),
                    'reason': 'Excess inventory with items expiring soon'
                })
        
        # Check upcoming menu needs (based on predicted demand)
        next_week = datetime.datetime.now().date() + datetime.timedelta(days=7)
        menu_suggestion = self.suggest_menu_for_sustainability(next_week)
        
        # Aggregate ingredient needs for suggested menu
        ingredient_needs = {}
        for menu_item in menu_suggestion['items']:
            recipe_id = menu_item['recipe_id']
            recipe = self.menu_planner.recipes[recipe_id]
            predicted_demand = menu_item['predicted_demand']
            
            for ingredient in recipe.ingredients:
                item_id = ingredient['item_id']
                quantity_needed = ingredient['quantity'] * predicted_demand
                
                if item_id not in ingredient_needs:
                    ingredient_needs[item_id] = 0
                ingredient_needs[item_id] += quantity_needed
        
        # Add forecast-based order recommendations
        for item_id, quantity_needed in ingredient_needs.items():
            if item_id in self.inventory_manager.inventory:
                item = self.inventory_manager.inventory[item_id]
                
                # If current quantity won't cover forecasted need
                if item.current_quantity < quantity_needed * 1.2:  # 20% buffer
                    additional_needed = (quantity_needed * 1.2) - item.current_quantity
                    if additional_needed > 0:
                        # Check if already in order list
                        existing_entry = next((x for x in recommendations['items_to_order'] 
                                            if x['item_id'] == item_id), None)
                        
                        if existing_entry:
                            # Update existing entry
                            new_quantity = max(existing_entry['recommended_order'], additional_needed)
                            existing_entry['recommended_order'] = new_quantity
                            existing_entry['estimated_cost'] = new_quantity * item.cost_per_unit
                            existing_entry['reason'] += ' and forecasted menu demand'
                            
                            # Update total cost
                            recommendations['total_estimated_cost'] += (new_quantity - existing_entry['recommended_order']) * item.cost_per_unit
                        else:
                            # Add new entry
                            recommendations['items_to_order'].append({
                                'item_id': item_id,
                                'name': item.name,
                                'current_quantity': item.current_quantity,
                                'recommended_order': additional_needed,
                                'unit': item.unit,
                                'estimated_cost': additional_needed * item.cost_per_unit,
                                'reason': 'Forecasted menu demand'
                            })
                            
                            recommendations['total_estimated_cost'] += additional_needed * item.cost_per_unit
        
        return recommendations

    def generate_daily_waste_diversion_plan(self):
        """Generate a plan for diverting potential food waste"""
        today = datetime.datetime.now().date()
        expiring_items = self.inventory_manager.get_expiring_items(3)  # Items expiring in 3 days
        
        if not expiring_items:
            return {"message": "No items expiring soon found"}
        
        diversion_plan = {
            'date': today,
            'expiring_items': len(expiring_items),
            'potential_waste_value': 0,
            'diversion_categories': {
                'donation': [],
                'staff_meal': [],
                'menu_special': [],
                'compost': []
            }
        }
        
        for item_id, details in expiring_items.items():
            item = self.inventory_manager.inventory[item_id]
            min_days = min(details['batches'].values())
            
            # Calculate estimated quantity and value
            quantity = len(details['batches'])  # Simplified
            value = quantity * item.cost_per_unit
            diversion_plan['potential_waste_value'] += value
            
            item_entry = {
                'item_id': item_id,
                'name': item.name,
                'quantity': quantity,
                'unit': item.unit,
                'days_until_expiry': min_days,
                'value': value
            }
            
            # Categorize based on item type and expiry
            if item.category in ['Meat', 'Seafood'] and min_days <= 1:
                # Staff meal for soon-expiring proteins
                diversion_plan['diversion_categories']['staff_meal'].append(item_entry)
            elif item.category in ['Produce', 'Dairy'] and min_days <= 2:
                # Menu special for fresh ingredients
                diversion_plan['diversion_categories']['menu_special'].append(item_entry)
            elif min_days >= 2 and item.category not in ['Prepared', 'Open']:
                # Donation for items with longer shelf life
                diversion_plan['diversion_categories']['donation'].append(item_entry)
            else:
                # Compost as last resort
                diversion_plan['diversion_categories']['compost'].append(item_entry)
        
        # Generate specific actions
        diversion_plan['recommended_actions'] = []
        
        # Donation actions
        if diversion_plan['diversion_categories']['donation']:
            donation_items = ', '.join([f"{item['quantity']} {item['unit']} {item['name']}" 
                                      for item in diversion_plan['diversion_categories']['donation'][:3]])
            diversion_plan['recommended_actions'].append(
                f"Contact food bank for donation pickup of: {donation_items}" + 
                (", and more" if len(diversion_plan['diversion_categories']['donation']) > 3 else "")
            )
        
        # Staff meal actions
        if diversion_plan['diversion_categories']['staff_meal']:
            staff_items = ', '.join([item['name'] for item in diversion_plan['diversion_categories']['staff_meal'][:3]])
            diversion_plan['recommended_actions'].append(
                f"Prepare staff meal utilizing: {staff_items}" + 
                (", and more" if len(diversion_plan['diversion_categories']['staff_meal']) > 3 else "")
            )
        
        # Menu special actions
        if diversion_plan['diversion_categories']['menu_special']:
            # Group by category
            by_category = {}
            for item in diversion_plan['diversion_categories']['menu_special']:
                category = self.inventory_manager.inventory[item['item_id']].category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(item)
            
            # Generate recipe ideas
            for category, items in by_category.items():
                if category == 'Produce':
                    item_names = ', '.join([item['name'] for item in items[:3]])
                    diversion_plan['recommended_actions'].append(
                        f"Create vegetable special using: {item_names}"
                    )
                elif category == 'Dairy':
                    item_names = ', '.join([item['name'] for item in items[:2]])
                    diversion_plan['recommended_actions'].append(
                        f"Create dairy-focused dessert using: {item_names}"
                    )
                elif category == 'Meat':
                    item_names = ', '.join([item['name'] for item in items[:2]])
                    diversion_plan['recommended_actions'].append(
                        f"Create protein-focused daily special using: {item_names}"
                    )
        
        return diversion_plan
    
    def run_demo(self):
        """Run a demonstration of the system's capabilities"""
        self.initialize_demo_data()
        
        demo_results = {
            'inventory_status': {
                'total_items': len(self.inventory_manager.inventory),
                'expiring_soon': len(self.inventory_manager.get_expiring_items(5))
            },
            'waste_analysis': self.waste_analyzer.identify_waste_patterns(),
            'menu_recommendations': self.menu_planner.generate_menu_recommendations(),
            'sustainability_menu': self.suggest_menu_for_sustainability(),
            'inventory_recommendations': self.generate_inventory_order_recommendations(),
            'waste_diversion_plan': self.generate_daily_waste_diversion_plan()
        }
        
        return demo_results


# Example usage of the system
if __name__ == "__main__":
    # Create and initialize the system
    system = AIFoodWasteSystem()
    demo_results = system.run_demo()
    
    # Display some results
    print("\n===== AI FOOD WASTE MANAGEMENT SYSTEM DEMO =====\n")
    
    print("INVENTORY STATUS:")
    print(f"Total inventory items: {demo_results['inventory_status']['total_items']}")
    print(f"Items expiring soon: {demo_results['inventory_status']['expiring_soon']}")
    
    print("\nWASTE ANALYSIS:")
    if 'top_reasons' in demo_results['waste_analysis']:
        for i, reason in enumerate(demo_results['waste_analysis']['top_reasons']):
            print(f"{i+1}. {reason['reason']}: ${reason['total_cost']:.2f} " +
                 f"({reason['percentage_of_total']:.1f}% of waste)")
    
    print("\nMENU RECOMMENDATIONS:")
    for i, rec in enumerate(demo_results['menu_recommendations']):
        print(f"{i+1}. {rec['action']} {rec['recipe_name']} - {rec['reason']}")
    
    print("\nSUSTAINABLE MENU SUGGESTION:")
    print(f"Waste reduction potential: ${demo_results['sustainability_menu']['waste_reduction_potential']:.2f}")
    print(f"Sustainability score: {demo_results['sustainability_menu']['sustainability_score']:.1f}/100")
    for item in demo_results['sustainability_menu']['items']:
        print(f"- {item['name']} (Predicted demand: {item['predicted_demand']} servings)")
    
    print("\nINVENTORY RECOMMENDATIONS:")
    print("Items to order:")
    for item in demo_results['inventory_recommendations']['items_to_order']:
        print(f"- {item['name']}: {item['recommended_order']} {item['unit']} (${item['estimated_cost']:.2f})")
    
    print("\nWASTE DIVERSION PLAN:")
    print(f"Potential waste value to be saved: ${demo_results['waste_diversion_plan']['potential_waste_value']:.2f}")
    print("Recommended actions:")
    for action in demo_results['waste_diversion_plan']['recommended_actions']:
        print(f"- {action}")
    
    print("\n===== END OF DEMO =====")
