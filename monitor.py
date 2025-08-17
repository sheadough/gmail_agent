import os
import re
import csv
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

# SMS (using Twilio - free trial available)
try:
    from twilio.rest import Client
except ImportError:
    print("Twilio not installed. SMS notifications will use textbelt only.")
    Client = None

# Alternative SMS option using textbelt (free but limited)
import requests

class TicketParser:
    """Parses Field Nation email content to extract ticket information"""
    
    def __init__(self):
        # Updated patterns to match actual Field Nation email structure
        self.patterns = {
            # Work Order ID - matches the link structure or table cell
            'work_order_id': r'workorders/(\d+)',
            
            # Company name and rating - from the h5 element
            'company_and_rating': r'<h5[^>]*>\s*([^<(]+?)\s*\(\s*([\d.]+)\s*<',
            
            # Location with distance - matches the specific format
            'location_distance': r'([^,]+,\s*[A-Z]{2}\s+\d{5})\s*\(\s*(\d+)\s*mi',
            
            # Pay information - hourly rate and max hours
            'hourly_pay': r'Hourly Rate:\s*\$(\d+(?:\.\d{2})?)/hour\s*\((\d+)\s*hours?\s*max\)',
            
            # Alternative pay pattern for flat rate
            'flat_pay': r'\$(\d+(?:\.\d{2})?)\s*(?:flat|total|fixed)',
            
            # Schedule information
            'schedule': r'([A-Za-z]+,\s*[A-Za-z]+\s+\d+\s*@\s*\d+:\d+\s*[AP]M)',
            
            # Work type
            'work_type': r'Type of Work[^>]*>\s*([^<]+)',
            
            # Industry
            'industry': r'Industry[^>]*>\s*([^<]+)',
            
            # Status (from image alt text or table)
            'status': r'alt="([^"]*)"[^>]*class="[^"]*status',
        }
    
    def parse_email(self, email_content: str) -> Dict:
        """Extract ticket information from Field Nation email content"""
        ticket_data = {
            'raw_content': email_content,
            'parsed_at': datetime.now().isoformat(),
        }
        
        # Clean up the content for better parsing
        content_clean = re.sub(r'\s+', ' ', email_content)  # Normalize whitespace
        content_lower = email_content.lower()
        
        # Extract Work Order ID
        wo_match = re.search(self.patterns['work_order_id'], email_content, re.IGNORECASE)
        if wo_match:
            ticket_data['ticket_id'] = wo_match.group(1)
            ticket_data['work_order_id'] = wo_match.group(1)
        
        # Extract Company and Rating
        company_rating_match = re.search(self.patterns['company_and_rating'], email_content, re.IGNORECASE | re.DOTALL)
        if company_rating_match:
            ticket_data['company'] = company_rating_match.group(1).strip()
            try:
                ticket_data['rating'] = float(company_rating_match.group(2))
            except ValueError:
                ticket_data['rating'] = None
        
        # Extract Location and Distance
        location_match = re.search(self.patterns['location_distance'], email_content, re.IGNORECASE)
        if location_match:
            ticket_data['location'] = location_match.group(1).strip()
            try:
                ticket_data['distance'] = float(location_match.group(2))
            except ValueError:
                ticket_data['distance'] = None
        
        # Extract Pay Information
        hourly_match = re.search(self.patterns['hourly_pay'], email_content, re.IGNORECASE)
        if hourly_match:
            try:
                ticket_data['hourly_pay'] = float(hourly_match.group(1))
                ticket_data['max_hours'] = float(hourly_match.group(2))
                ticket_data['calculated_total_pay'] = ticket_data['hourly_pay'] * ticket_data['max_hours']
                ticket_data['total_hours'] = ticket_data['max_hours']
            except ValueError:
                pass
        else:
            # Try flat rate pattern
            flat_match = re.search(self.patterns['flat_pay'], email_content, re.IGNORECASE)
            if flat_match:
                try:
                    ticket_data['calculated_total_pay'] = float(flat_match.group(1))
                    ticket_data['hourly_pay'] = None  # Flat rate, not hourly
                    ticket_data['max_hours'] = None
                except ValueError:
                    pass
        
        # Extract Schedule
        schedule_match = re.search(self.patterns['schedule'], email_content, re.IGNORECASE)
        if schedule_match:
            ticket_data['schedule'] = schedule_match.group(1).strip()
        
        # Extract Work Type
        work_type_match = re.search(self.patterns['work_type'], email_content, re.IGNORECASE | re.DOTALL)
        if work_type_match:
            ticket_data['work_type'] = work_type_match.group(1).strip()
        
        # Extract Industry
        industry_match = re.search(self.patterns['industry'], email_content, re.IGNORECASE | re.DOTALL)
        if industry_match:
            ticket_data['industry'] = industry_match.group(1).strip()
        
        # Extract Status
        status_match = re.search(self.patterns['status'], email_content, re.IGNORECASE)
        if status_match:
            ticket_data['status'] = status_match.group(1).strip()
        else:
            # Default status extraction from common patterns
            if 'available' in content_lower:
                ticket_data['status'] = 'Available'
            elif 'assigned' in content_lower:
                ticket_data['status'] = 'Assigned'
            else:
                ticket_data['status'] = 'Unknown'
        
        # Set default values for missing fields
        default_fields = {
            'ticket_id': 'Unknown',
            'work_order_id': 'Unknown',
            'company': 'Unknown Company',
            'rating': None,
            'location': 'Unknown Location',
            'distance': None,
            'hourly_pay': None,
            'max_hours': None,
            'calculated_total_pay': None,
            'total_hours': None,
            'schedule': 'Unknown Schedule',
            'work_type': 'Unknown',
            'industry': 'Unknown',
            'status': 'Unknown'
        }
        
        for field, default_value in default_fields.items():
            if field not in ticket_data or ticket_data[field] is None:
                ticket_data[field] = default_value
        
        # Extract urgency indicators
        urgency_words = ['urgent', 'asap', 'emergency', 'rush', 'immediate', 'critical', 'priority']
        ticket_data['urgency_score'] = sum(1 for word in urgency_words if word in content_lower)
        
        # Extract quality indicators
        quality_indicators = {
            'has_equipment_list': bool(re.search(r'required equipment|tools|ppe', content_lower)),
            'has_ppe_requirements': bool(re.search(r'ppe|hard hat|safety|steel toe', content_lower)),
            'company_rated': ticket_data.get('rating') is not None,
            'has_detailed_scope': bool(re.search(r'scope of work|tasks include|responsibilities', content_lower)),
            'has_schedule_info': ticket_data.get('schedule') != 'Unknown Schedule',
            'has_location_info': ticket_data.get('location') != 'Unknown Location'
        }
        
        ticket_data['quality_score'] = sum(quality_indicators.values())
        
        # Debug information
        ticket_data['parsing_debug'] = {
            'found_work_order': bool(wo_match),
            'found_company': bool(company_rating_match),
            'found_location': bool(location_match),
            'found_pay': bool(hourly_match or flat_match),
            'found_schedule': bool(schedule_match),
            'content_length': len(email_content),
            'quality_indicators': quality_indicators
        }
        
        return ticket_data

class GmailMonitor:
    """Handles Gmail API interactions"""
    
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, credentials_file='credentials.json'):
        self.credentials_file = credentials_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Token refresh failed: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"Credentials file {self.credentials_file} not found")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        self.service = build('gmail', 'v1', credentials=creds)
    
    def get_recent_emails(self, hours_back=24, sender='support@example.com'):
        """Get emails from specific sender within time range"""
        try:
            # Calculate timestamp for query
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            timestamp = int(time_threshold.timestamp())
            
            query = f'from:{sender} after:{timestamp}'
            
            results = self.service.users().messages().list(
                userId='me', q=query, maxResults=50).execute()
            
            messages = results.get('messages', [])
            
            emails = []
            for message in messages:
                try:
                    msg = self.service.users().messages().get(
                        userId='me', id=message['id'], format='full').execute()
                    
                    # Decode email content
                    email_content = self._extract_email_body(msg['payload'])
                    
                    # Get timestamp
                    timestamp = int(msg['internalDate'])
                    dt = datetime.fromtimestamp(timestamp / 1000)
                    
                    emails.append({
                        'id': message['id'],
                        'content': email_content,
                        'timestamp': dt.isoformat(),
                        'raw_timestamp': timestamp
                    })
                except Exception as e:
                    print(f"Error processing message {message['id']}: {e}")
                    continue
            
            return emails
        except Exception as e:
            print(f"Error fetching emails: {e}")
            return []
    
    def _extract_email_body(self, payload):
        """Extract email body from payload, prioritizing HTML content"""
        body = ""
        
        def extract_from_part(part):
            if 'parts' in part:
                # Multi-part message
                html_body = ""
                text_body = ""
                
                for subpart in part['parts']:
                    if subpart['mimeType'] == 'text/html' and 'data' in subpart.get('body', {}):
                        data = subpart['body']['data']
                        html_body = base64.urlsafe_b64decode(data).decode('utf-8')
                    elif subpart['mimeType'] == 'text/plain' and 'data' in subpart.get('body', {}):
                        data = subpart['body']['data']
                        text_body = base64.urlsafe_b64decode(data).decode('utf-8')
                    else:
                        # Recursive for nested parts
                        nested_content = extract_from_part(subpart)
                        if nested_content:
                            if 'html' in subpart.get('mimeType', '').lower():
                                html_body += nested_content
                            else:
                                text_body += nested_content
                
                # Prefer HTML for emails as they contain structured data
                return html_body if html_body else text_body
            else:
                # Single part message
                if 'data' in payload.get('body', {}):
                    data = payload['body']['data']
                    return base64.urlsafe_b64decode(data).decode('utf-8')
            return ""
        
        body = extract_from_part(payload)
        
        return body

#EscalationEngine, SMSNotifier, and Config classes
class EscalationEngine:
    """Handles ticket escalation logic and ML learning"""
    
    def __init__(self, csv_file='ticket_data.csv'):
        self.csv_file = csv_file
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = ['hourly_pay', 'max_hours', 'calculated_total_pay', 'distance', 
                               'urgency_score', 'quality_score', 'rating']
        self.is_trained = False
        self._ensure_csv_exists()
        self._load_or_train_model()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            headers = ['ticket_id', 'timestamp', 'hourly_pay', 'max_hours', 'calculated_total_pay', 
                      'distance', 'urgency_score', 'quality_score', 'rating', 'location', 
                      'company', 'work_type', 'escalated', 'usefulness_score', 'raw_content']
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
    
    def should_escalate_baseline(self, ticket_data: Dict, baseline_params: Dict) -> bool:
        """Determine if ticket should be escalated based on baseline parameters"""
        criteria_met = 0
        total_criteria = 0
        
        def safe_get_number(data, key, default=0):
            value = data.get(key)
            if value is None or value == 'Unknown' or value == '':
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Check minimum hourly pay
        if baseline_params.get('min_hourly_pay') is not None:
            total_criteria += 1
            hourly_pay = safe_get_number(ticket_data, 'hourly_pay', 0)
            if hourly_pay >= baseline_params['min_hourly_pay']:
                criteria_met += 1
        
        # Check maximum distance
        if baseline_params.get('max_distance') is not None:
            total_criteria += 1
            distance = safe_get_number(ticket_data, 'distance', float('inf'))
            if distance <= baseline_params['max_distance']:
                criteria_met += 1
        
        # Check minimum total pay
        if baseline_params.get('min_total_pay') is not None:
            total_criteria += 1
            total_pay = safe_get_number(ticket_data, 'calculated_total_pay', 0)
            if total_pay >= baseline_params['min_total_pay']:
                criteria_met += 1
        
        # Check maximum hours
        if baseline_params.get('max_hours') is not None:
            total_criteria += 1
            max_hours = safe_get_number(ticket_data, 'max_hours', float('inf'))
            if max_hours <= baseline_params['max_hours']:
                criteria_met += 1
        
        # Require meeting threshold percentage of criteria
        if total_criteria == 0:
            return False
        
        threshold = baseline_params.get('criteria_threshold', 0.7)
        result = (criteria_met / total_criteria) >= threshold
        
        return result
    
    def predict_usefulness(self, ticket_data: Dict) -> float:
        """Predict ticket usefulness using ML model"""
        if not self.is_trained:
            return 5.0  # Default neutral score
        
        try:
            features = self._extract_features(ticket_data)
            if features is None:
                return 5.0
            
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            return max(1.0, min(10.0, prediction))  # Clamp between 1-10
        except Exception as e:
            print(f"Error predicting usefulness: {e}")
            return 5.0
    
    def _extract_features(self, ticket_data: Dict) -> Optional[List[float]]:
        """Extract numerical features for ML model"""
        try:
            features = []
            for col in self.feature_columns:
                value = ticket_data.get(col, 0)
                if value is None or value == 'Unknown' or value == '':
                    value = 0
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    features.append(0.0)
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        model_file = 'escalation_model.pkl'
        scaler_file = 'feature_scaler.pkl'
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.is_trained = True
                print("Loaded existing ML model")
                return
            except Exception as e:
                print(f"Error loading model: {e}")
        
        self._train_model()
    
    def _train_model(self):
        """Train the ML model with existing data"""
        try:
            if not os.path.exists(self.csv_file):
                return
                
            df = pd.read_csv(self.csv_file)
            
            # Need at least 10 samples with usefulness scores to train
            training_data = df[df['usefulness_score'].notna() & (df['usefulness_score'] > 0)]
            
            if len(training_data) < 10:
                print(f"Not enough training data ({len(training_data)} samples). Need at least 10.")
                return
            
            # Prepare features
            X = []
            y = []
            
            for _, row in training_data.iterrows():
                features = self._extract_features(row.to_dict())
                if features is not None:
                    X.append(features)
                    y.append(row['usefulness_score'])
            
            if len(X) < 10:
                print("Not enough valid feature vectors for training")
                return
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Save model
            joblib.dump(self.model, 'escalation_model.pkl')
            joblib.dump(self.scaler, 'feature_scaler.pkl')
            
            print(f"Model trained with {len(X)} samples")
            
        except Exception as e:
            print(f"Error training model: {e}")
    
    def save_ticket_data(self, ticket_data: Dict, escalated: bool, usefulness_score: Optional[int] = None):
        """Save ticket data to CSV"""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    ticket_data.get('ticket_id', ''),
                    datetime.now().isoformat(),
                    ticket_data.get('hourly_pay'),
                    ticket_data.get('max_hours'),
                    ticket_data.get('calculated_total_pay'),
                    ticket_data.get('distance'),
                    ticket_data.get('urgency_score', 0),
                    ticket_data.get('quality_score', 0),
                    ticket_data.get('rating'),
                    ticket_data.get('location', ''),
                    ticket_data.get('company', ''),
                    ticket_data.get('work_type', ''),
                    escalated,
                    usefulness_score,
                    ticket_data.get('raw_content', '')[:500] if ticket_data.get('raw_content') else ''
                ])
        except Exception as e:
            print(f"Error saving ticket data: {e}")
    
    def update_usefulness_score(self, ticket_id: str, score: int):
        """Update usefulness score for a specific ticket"""
        try:
            if not os.path.exists(self.csv_file):
                return False
                
            df = pd.read_csv(self.csv_file)
            
            # Update score
            mask = df['ticket_id'] == ticket_id
            if mask.any():
                df.loc[mask, 'usefulness_score'] = score
                df.to_csv(self.csv_file, index=False)
                
                # Retrain model with new data
                self._train_model()
                
                return True
            return False
        except Exception as e:
            print(f"Error updating usefulness score: {e}")
            return False

class SMSNotifier:
    """Handles SMS notifications"""
    
    def __init__(self, service='textbelt', config=None):
        self.service = service
        self.config = config or {}
        
        if service == 'twilio' and config and Client:
            self.twilio_client = Client(config['account_sid'], config['auth_token'])
        else:
            self.twilio_client = None
    
    def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification"""
        try:
            if self.service == 'twilio' and self.twilio_client:
                message = self.twilio_client.messages.create(
                    body=message,
                    from_=self.config['from_number'],
                    to=phone_number
                )
                return True
            
            elif self.service == 'textbelt':
                # Free service with daily limits
                resp = requests.post('https://textbelt.com/text', {
                    'phone': phone_number,
                    'message': message,
                    'key': self.config.get('api_key', 'textbelt')
                })
                return resp.json().get('success', False)
            
            return False
        except Exception as e:
            print(f"SMS send error: {e}")
            return False
    
    def format_ticket_message(self, ticket_data: Dict, predicted_score: float = None) -> str:
        """Format ticket data into SMS message"""
        message_parts = ["🎫 Field Nation Alert"]
        
        ticket_id = ticket_data.get('ticket_id', ticket_data.get('work_order_id'))
        if ticket_id and ticket_id != 'Unknown':
            message_parts.append(f"WO: {ticket_id}")
        
        company = ticket_data.get('company')
        if company and company != 'Unknown Company':
            company = company[:20]
            rating = ticket_data.get('rating')
            if rating:
                message_parts.append(f"{company} ({rating}★)")
            else:
                message_parts.append(f"{company}")
        
        # Pay information
        hourly_pay = ticket_data.get('hourly_pay')
        max_hours = ticket_data.get('max_hours')
        total_pay = ticket_data.get('calculated_total_pay')
        
        if hourly_pay and max_hours:
            pay_str = f"${hourly_pay}/hr ({max_hours}h max)"
            message_parts.append(pay_str)
        elif total_pay:
            message_parts.append(f"💵 ${total_pay}")
        
        distance = ticket_data.get('distance')
        if distance:
            message_parts.append(f"📍 {distance} mi")
        
        location = ticket_data.get('location')
        if location and location != 'Unknown Location':
            # Extract just city and state for brevity
            location_short = re.sub(r',?\s*\d{5}.*', '', location)[:25]
            message_parts.append(f"📌 {location_short}")
        
        if predicted_score:
            message_parts.append(f"🤖 AI: {predicted_score:.1f}/10")
        
        return " | ".join(message_parts)

class Config:
    """Application configuration"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            'baseline_params': {
                'min_hourly_pay': 25.0,
                'max_distance': 30.0,
                'min_total_pay': 100.0,
                'criteria_threshold': 0.7
            },
            'sms': {
                'service': 'textbelt',
                'phone_number': '',
                'api_key': 'textbelt'
            },
            'monitoring': {
                'check_interval_minutes': 1,
                'hours_back': 2
            },
            'ml_threshold': 7.0
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        self.data = default_config
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            value = value.get(k, default if k == keys[-1] else {})
        return value
    
    def set(self, key, value):
        """Set configuration value"""
        keys = key.split('.')
        config_ref = self.data
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        config_ref[keys[-1]] = value
        self.save_config()

if __name__ == "__main__":
    print("Field Nation Monitor - Core Components Loaded")
    print("Run 'python app.py' to start the Flask dashboard")