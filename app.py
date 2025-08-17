from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from monitor import (
    TicketParser, GmailMonitor, EscalationEngine, 
    Config, SMSNotifier
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Initialize components
config = Config()
parser = TicketParser()
escalation_engine = EscalationEngine()

# Global variables for monitoring
monitoring_active = False
monitoring_thread = None
last_check_time = None
recent_tickets = []

def monitoring_loop():
    """Background monitoring loop"""
    global monitoring_active, last_check_time, recent_tickets
    
    gmail_monitor = None
    sms_notifier = None
    
    try:
        gmail_monitor = GmailMonitor()
        
        # Initialize SMS if configured
        sms_config = config.get('sms', {})
        if sms_config.get('phone_number'):
            sms_notifier = SMSNotifier(
                service=sms_config.get('service', 'textbelt'),
                config=sms_config
            )
    except Exception as e:
        print(f"Error initializing monitors: {e}")
        return
    
    while monitoring_active:
        try:
            print(f"Checking emails at {datetime.now()}")
            last_check_time = datetime.now()
            
            # Get recent emails
            hours_back = config.get('monitoring.hours_back', 12)
            emails = gmail_monitor.get_recent_emails(hours_back=hours_back)
            
            for email in emails:
                # Parse ticket data
                ticket_data = parser.parse_email(email['content'])
                ticket_data['email_id'] = email['id']
                ticket_data['timestamp'] = email['timestamp']
                
                # Check if already processed (simple duplicate prevention)
                if any(t.get('email_id') == email['id'] for t in recent_tickets):
                    continue
                
                # Determine if should escalate based on baseline
                baseline_params = config.get('baseline_params', {})
                should_escalate_baseline = escalation_engine.should_escalate_baseline(
                    ticket_data, baseline_params
                )
                
                # Get ML prediction
                predicted_usefulness = escalation_engine.predict_usefulness(ticket_data)
                
                # Final escalation decision (baseline OR high ML score)
                should_escalate = (should_escalate_baseline or 
                                 predicted_usefulness >= config.get('ml_threshold', 7.0))
                
                # Add to recent tickets
                ticket_data['should_escalate'] = should_escalate
                ticket_data['predicted_usefulness'] = predicted_usefulness
                ticket_data['baseline_match'] = should_escalate_baseline
                recent_tickets.append(ticket_data)
                
                # Keep only last 50 tickets
                if len(recent_tickets) > 50:
                    recent_tickets = recent_tickets[-50:]
                
                # Save to CSV
                escalation_engine.save_ticket_data(ticket_data, should_escalate)
                
                # Send SMS if escalating and SMS is configured
                if should_escalate and sms_notifier:
                    message = sms_notifier.format_ticket_message(
                        ticket_data, predicted_usefulness
                    )
                    success = sms_notifier.send_sms(
                        config.get('sms.phone_number'), message
                    )
                    print(f"SMS sent: {success} for ticket {ticket_data.get('ticket_id', 'Unknown')}")
            
            # Wait for next check
            interval_minutes = config.get('monitoring.check_interval_minutes', 15)
            time.sleep(interval_minutes * 60)
            
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(60)  # Wait 1 minute on error

@app.route('/')
def dashboard():
    """Main dashboard page"""
    # Get recent ticket stats
    escalated_count = sum(1 for t in recent_tickets if t.get('should_escalate'))
    total_count = len(recent_tickets)
    
    # Get configuration
    baseline_params = config.get('baseline_params', {})
    sms_config = config.get('sms', {})
    
    return render_template('dashboard.html',
                         monitoring_active=monitoring_active,
                         last_check_time=last_check_time,
                         escalated_count=escalated_count,
                         total_count=total_count,
                         recent_tickets=recent_tickets[:10],  # Show last 10
                         baseline_params=baseline_params,
                         sms_configured=bool(sms_config.get('phone_number')),
                         ml_trained=escalation_engine.is_trained)

@app.route('/api/tickets')
def api_tickets():
    """API endpoint for ticket data"""
    return jsonify({
        'tickets': recent_tickets,
        'stats': {
            'total': len(recent_tickets),
            'escalated': sum(1 for t in recent_tickets if t.get('should_escalate')),
            'last_check': last_check_time.isoformat() if last_check_time else None
        }
    })

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """API endpoint for configuration"""
    if request.method == 'POST':
        data = request.json
        
        # Update baseline parameters
        if 'baseline_params' in data:
            for key, value in data['baseline_params'].items():
                config.set(f'baseline_params.{key}', value)
        
        # Update SMS configuration
        if 'sms' in data:
            for key, value in data['sms'].items():
                config.set(f'sms.{key}', value)
        
        # Update monitoring configuration
        if 'monitoring' in data:
            for key, value in data['monitoring'].items():
                config.set(f'monitoring.{key}', value)
        
        return jsonify({'status': 'success'})
    
    return jsonify(config.data)

@app.route('/api/monitoring/<action>')
def api_monitoring(action):
    """API endpoint for monitoring control"""
    global monitoring_active, monitoring_thread
    
    if action == 'start':
        if not monitoring_active:
            monitoring_active = True
            monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            monitoring_thread.start()
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})
    
    elif action == 'stop':
        monitoring_active = False
        return jsonify({'status': 'stopped'})
    
    return jsonify({'status': 'unknown_action'})

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API endpoint for feedback on ticket usefulness"""
    data = request.json
    ticket_id = data.get('ticket_id')
    usefulness_score = data.get('usefulness_score')
    
    if not ticket_id or not usefulness_score:
        return jsonify({'error': 'Missing ticket_id or usefulness_score'}), 400
    
    success = escalation_engine.update_usefulness_score(ticket_id, usefulness_score)
    
    return jsonify({'success': success})

@app.route('/api/test_sms', methods=['POST'])
def api_test_sms():
    """Test SMS functionality"""
    sms_config = config.get('sms', {})
    
    if not sms_config.get('phone_number'):
        return jsonify({'error': 'Phone number not configured'}), 400
    
    sms_notifier = SMSNotifier(
        service=sms_config.get('service', 'textbelt'),
        config=sms_config
    )
    
    test_message = "Field Nation Monitor - Test message sent successfully! 🎫"
    success = sms_notifier.send_sms(sms_config['phone_number'], test_message)
    
    return jsonify({'success': success})

@app.route('/analytics')
def analytics():
    """Analytics page"""
    try:
        # Load ticket data from CSV
        df = pd.read_csv('ticket_data.csv')
        
        # Basic analytics
        total_tickets = len(df)
        escalated_tickets = len(df[df['escalated'] == True])
        
        # ML model performance
        rated_tickets = df[df['usefulness_score'].notna() & (df['usefulness_score'] > 0)]
        avg_usefulness = rated_tickets['usefulness_score'].mean() if len(rated_tickets) > 0 else 0
        
        # Recent activity (last 7 days)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent_df = df[df['timestamp'] >= datetime.now() - timedelta(days=7)]
            recent_escalated = len(recent_df[recent_df['escalated'] == True])
        else:
            recent_escalated = 0
        
        analytics_data = {
            'total_tickets': total_tickets,
            'escalated_tickets': escalated_tickets,
            'escalation_rate': (escalated_tickets / max(total_tickets, 1)) * 100,
            'avg_usefulness': round(avg_usefulness, 2),
            'rated_tickets': len(rated_tickets),
            'recent_escalated': recent_escalated,
            'ml_trained': escalation_engine.is_trained
        }
        
        return render_template('analytics.html', **analytics_data)
        
    except Exception as e:
        return render_template('analytics.html', 
                             error=f"Error loading analytics: {e}",
                             total_tickets=0,
                             escalated_tickets=0,
                             escalation_rate=0,
                             avg_usefulness=0,
                             rated_tickets=0,
                             recent_escalated=0,
                             ml_trained=False)

@app.route('/tickets')
def tickets():
    """Tickets history page"""
    try:
        df = pd.read_csv('ticket_data.csv')
        df = df.fillna('')  # Replace NaN with empty strings for display
        
        # Convert timestamp to readable format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Sort by timestamp descending
        df = df.sort_values('timestamp', ascending=False)
        
        # Convert to list of dictionaries for template
        tickets_data = df.head(100).to_dict('records')  # Show last 100 tickets
        
        return render_template('tickets.html', tickets=tickets_data)
        
    except Exception as e:
        return render_template('tickets.html', error=f"Error loading tickets: {e}", tickets=[])

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html', config=config.data)

if __name__ == '__main__':
    print("Gmail Monitor Dashboard")
    print("Starting Flask app on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)