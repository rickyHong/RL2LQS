import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


@dataclass
class EmployeeProfile:
    """Employee profile containing all relevant HR information"""
    employee_id: int
    skills: np.ndarray
    performance_history: List[float]
    satisfaction_score: float
    engagement_level: float
    productivity_index: float
    learning_ability: float
    team_collaboration: float
    leadership_potential: float
    work_life_balance: float
    career_stage: str
    department: str
    tenure: int
    

@dataclass
class HRMetrics:
    """Key HR performance indicators"""
    employee_satisfaction: float
    employee_engagement: float
    productivity_index: float
    retention_rate: float
    recruitment_quality: float
    training_effectiveness: float
    organizational_culture_score: float
    leadership_effectiveness: float
    innovation_index: float
    cost_per_hire: float
    time_to_hire: float
    absenteeism_rate: float
    turnover_rate: float


class HRMEnvironment:
    """
    Human Resource Management Environment for reinforcement learning.
    Simulates organizational dynamics and HR decision outcomes.
    """
    
    def __init__(self, **env_params):
        # Initialize environment parameters
        self.env_params = env_params
        self.num_employees = env_params.get('num_employees', 1000)
        self.num_departments = env_params.get('num_departments', 10)
        self.skill_dimensions = env_params.get('skill_dimensions', 20)
        self.time_horizon = env_params.get('time_horizon', 12)  # months
        
        # Initialize state variables
        self.current_time = 0
        self.employees: List[EmployeeProfile] = []
        self.department_budgets = np.random.uniform(50000, 500000, self.num_departments)
        self.company_goals = self._generate_company_goals()
        
        # Initialize scalers for normalization
        self.scaler = StandardScaler()
        
        # Initialize HR metrics tracking
        self.hr_metrics_history = []
        self.performance_targets = self._set_performance_targets()
        
        # Market conditions and external factors
        self.market_conditions = {
            'economic_growth': np.random.uniform(0.8, 1.2),
            'industry_competitiveness': np.random.uniform(0.7, 1.3),
            'technology_advancement': np.random.uniform(0.9, 1.4),
            'regulatory_changes': np.random.uniform(0.8, 1.1)
        }
        
        # Initialize employee population
        self._generate_initial_employees()
        
        # Action and observation spaces
        self.action_space_size = self._calculate_action_space_size()
        self.observation_space_size = self._calculate_observation_space_size()
        
    def _generate_company_goals(self) -> Dict:
        """Generate company strategic goals"""
        return {
            'revenue_growth': np.random.uniform(0.05, 0.25),
            'market_share_increase': np.random.uniform(0.02, 0.15),
            'customer_satisfaction': np.random.uniform(0.8, 0.95),
            'innovation_rate': np.random.uniform(0.1, 0.4),
            'sustainability_score': np.random.uniform(0.6, 0.9)
        }
    
    def _set_performance_targets(self) -> Dict:
        """Set target values for HR metrics"""
        return {
            'employee_satisfaction': 0.85,
            'employee_engagement': 0.80,
            'productivity_index': 1.2,
            'retention_rate': 0.90,
            'recruitment_quality': 0.85,
            'training_effectiveness': 0.80,
            'organizational_culture_score': 0.85,
            'leadership_effectiveness': 0.80,
            'innovation_index': 0.75,
            'turnover_rate': 0.10,
            'absenteeism_rate': 0.05
        }
    
    def _generate_initial_employees(self):
        """Generate initial employee population"""
        departments = [f"Dept_{i}" for i in range(self.num_departments)]
        career_stages = ["Junior", "Mid-level", "Senior", "Executive"]
        
        for i in range(self.num_employees):
            employee = EmployeeProfile(
                employee_id=i,
                skills=np.random.uniform(0.3, 1.0, self.skill_dimensions),
                performance_history=[np.random.uniform(0.6, 1.0) for _ in range(12)],
                satisfaction_score=np.random.uniform(0.4, 0.9),
                engagement_level=np.random.uniform(0.3, 0.95),
                productivity_index=np.random.uniform(0.7, 1.3),
                learning_ability=np.random.uniform(0.5, 1.0),
                team_collaboration=np.random.uniform(0.4, 0.95),
                leadership_potential=np.random.uniform(0.2, 0.9),
                work_life_balance=np.random.uniform(0.4, 0.9),
                career_stage=np.random.choice(career_stages),
                department=np.random.choice(departments),
                tenure=np.random.randint(1, 120)  # months
            )
            self.employees.append(employee)
    
    def _calculate_action_space_size(self) -> int:
        """Calculate the size of action space for HR decisions"""
        # Actions include: hiring, training, promotion, retention programs, etc.
        actions = {
            'recruitment_strategy': 5,  # Different recruitment approaches
            'training_programs': 8,     # Various training types
            'compensation_adjustments': 6,  # Salary and benefit changes
            'team_restructuring': 4,    # Organizational changes
            'performance_management': 5,  # Performance review strategies
            'culture_initiatives': 6,   # Culture improvement programs
            'technology_investments': 4,  # HR tech implementations
            'leadership_development': 5   # Leadership programs
        }
        return sum(actions.values())
    
    def _calculate_observation_space_size(self) -> int:
        """Calculate observation space size"""
        # Observations include current HR metrics, employee statistics, market conditions
        return (
            13 +  # HR metrics
            self.skill_dimensions +  # Average skills
            10 +  # Employee statistics
            4 +   # Market conditions
            5     # Company goals
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_time = 0
        self.hr_metrics_history = []
        self._generate_initial_employees()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state observation"""
        # Calculate current HR metrics
        current_metrics = self._calculate_hr_metrics()
        
        # Employee statistics
        avg_skills = np.mean([emp.skills for emp in self.employees], axis=0)
        avg_satisfaction = np.mean([emp.satisfaction_score for emp in self.employees])
        avg_engagement = np.mean([emp.engagement_level for emp in self.employees])
        avg_productivity = np.mean([emp.productivity_index for emp in self.employees])
        avg_learning = np.mean([emp.learning_ability for emp in self.employees])
        avg_collaboration = np.mean([emp.team_collaboration for emp in self.employees])
        avg_leadership = np.mean([emp.leadership_potential for emp in self.employees])
        avg_work_life_balance = np.mean([emp.work_life_balance for emp in self.employees])
        avg_tenure = np.mean([emp.tenure for emp in self.employees])
        employee_count = len(self.employees)
        
        # Construct state vector
        state = np.concatenate([
            [
                current_metrics.employee_satisfaction,
                current_metrics.employee_engagement,
                current_metrics.productivity_index,
                current_metrics.retention_rate,
                current_metrics.recruitment_quality,
                current_metrics.training_effectiveness,
                current_metrics.organizational_culture_score,
                current_metrics.leadership_effectiveness,
                current_metrics.innovation_index,
                current_metrics.cost_per_hire,
                current_metrics.time_to_hire,
                current_metrics.absenteeism_rate,
                current_metrics.turnover_rate
            ],
            avg_skills,
            [
                avg_satisfaction,
                avg_engagement,
                avg_productivity,
                avg_learning,
                avg_collaboration,
                avg_leadership,
                avg_work_life_balance,
                avg_tenure,
                employee_count,
                self.current_time
            ],
            list(self.market_conditions.values()),
            list(self.company_goals.values())
        ])
        
        return state.astype(np.float32)
    
    def _calculate_hr_metrics(self) -> HRMetrics:
        """Calculate current HR metrics based on employee data"""
        if not self.employees:
            return HRMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Employee satisfaction and engagement
        satisfaction = np.mean([emp.satisfaction_score for emp in self.employees])
        engagement = np.mean([emp.engagement_level for emp in self.employees])
        
        # Productivity metrics
        productivity = np.mean([emp.productivity_index for emp in self.employees])
        
        # Retention and turnover (simulated based on satisfaction and engagement)
        retention_probability = 0.5 + 0.3 * satisfaction + 0.2 * engagement
        retention_rate = min(0.98, max(0.70, retention_probability))
        turnover_rate = 1 - retention_rate
        
        # Quality metrics (based on skills and performance)
        avg_skills = np.mean([np.mean(emp.skills) for emp in self.employees])
        recruitment_quality = min(1.0, avg_skills * 1.2)
        
        # Training effectiveness (based on learning ability and skill development)
        learning_avg = np.mean([emp.learning_ability for emp in self.employees])
        training_effectiveness = min(1.0, learning_avg * 0.9 + 0.1)
        
        # Organizational culture (based on collaboration and work-life balance)
        collaboration_avg = np.mean([emp.team_collaboration for emp in self.employees])
        work_life_avg = np.mean([emp.work_life_balance for emp in self.employees])
        culture_score = (collaboration_avg + work_life_avg) / 2
        
        # Leadership effectiveness
        leadership_avg = np.mean([emp.leadership_potential for emp in self.employees])
        leadership_effectiveness = leadership_avg * 0.85 + 0.15
        
        # Innovation index (based on skills, learning, and leadership)
        innovation_index = (avg_skills + learning_avg + leadership_avg) / 3 * 0.8
        
        # Cost and time metrics (simulated)
        cost_per_hire = np.random.uniform(3000, 15000)
        time_to_hire = np.random.uniform(20, 60)  # days
        
        # Absenteeism (inversely related to satisfaction and work-life balance)
        absenteeism_rate = max(0.01, 0.15 - 0.1 * (satisfaction + work_life_avg) / 2)
        
        return HRMetrics(
            employee_satisfaction=satisfaction,
            employee_engagement=engagement,
            productivity_index=productivity,
            retention_rate=retention_rate,
            recruitment_quality=recruitment_quality,
            training_effectiveness=training_effectiveness,
            organizational_culture_score=culture_score,
            leadership_effectiveness=leadership_effectiveness,
            innovation_index=innovation_index,
            cost_per_hire=cost_per_hire,
            time_to_hire=time_to_hire,
            absenteeism_rate=absenteeism_rate,
            turnover_rate=turnover_rate
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment and return next state, reward, done, info
        
        Args:
            action: Integer representing HR action to take
            
        Returns:
            next_state: Next state observation
            reward: Reward for the action
            done: Whether episode is finished
            info: Additional information
        """
        # Decode action
        action_effects = self._decode_action(action)
        
        # Apply action effects to employees
        self._apply_action_effects(action_effects)
        
        # Update time
        self.current_time += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_time >= self.time_horizon
        
        # Get new state
        next_state = self.get_state()
        
        # Update metrics history
        current_metrics = self._calculate_hr_metrics()
        self.hr_metrics_history.append(current_metrics)
        
        # Additional info
        info = {
            'current_metrics': current_metrics,
            'action_effects': action_effects,
            'time_step': self.current_time
        }
        
        return next_state, reward, done, info
    
    def _decode_action(self, action: int) -> Dict:
        """Decode integer action into specific HR interventions"""
        # Define action mappings
        actions = {
            'recruitment_strategy': 5,
            'training_programs': 8,
            'compensation_adjustments': 6,
            'team_restructuring': 4,
            'performance_management': 5,
            'culture_initiatives': 6,
            'technology_investments': 4,
            'leadership_development': 5
        }
        
        # Decode action
        action_type = None
        action_value = action
        cumulative = 0
        
        for act_name, act_size in actions.items():
            if action_value < cumulative + act_size:
                action_type = act_name
                action_value = action_value - cumulative
                break
            cumulative += act_size
        
        return {
            'type': action_type,
            'value': action_value,
            'intensity': np.random.uniform(0.1, 0.3)  # Effect intensity
        }
    
    def _apply_action_effects(self, action_effects: Dict):
        """Apply the effects of HR actions to employee population"""
        action_type = action_effects['type']
        intensity = action_effects['intensity']
        
        for employee in self.employees:
            if action_type == 'recruitment_strategy':
                # Improve recruitment quality affects new hires (simulated)
                if np.random.random() < 0.1:  # 10% chance of new hire effect
                    employee.skills *= (1 + intensity * 0.1)
                    
            elif action_type == 'training_programs':
                # Training improves skills and learning ability
                skill_boost = intensity * employee.learning_ability * 0.05
                employee.skills = np.minimum(employee.skills + skill_boost, 1.0)
                employee.learning_ability = min(1.0, employee.learning_ability + intensity * 0.02)
                
            elif action_type == 'compensation_adjustments':
                # Compensation affects satisfaction and retention
                employee.satisfaction_score = min(1.0, employee.satisfaction_score + intensity * 0.1)
                
            elif action_type == 'team_restructuring':
                # Team changes affect collaboration
                if np.random.random() < 0.3:  # 30% of employees affected
                    employee.team_collaboration += np.random.uniform(-intensity*0.05, intensity*0.1)
                    employee.team_collaboration = np.clip(employee.team_collaboration, 0.1, 1.0)
                    
            elif action_type == 'performance_management':
                # Performance management affects productivity and engagement
                employee.productivity_index *= (1 + intensity * 0.05)
                employee.engagement_level = min(1.0, employee.engagement_level + intensity * 0.03)
                
            elif action_type == 'culture_initiatives':
                # Culture programs affect work-life balance and satisfaction
                employee.work_life_balance = min(1.0, employee.work_life_balance + intensity * 0.08)
                employee.satisfaction_score = min(1.0, employee.satisfaction_score + intensity * 0.05)
                
            elif action_type == 'technology_investments':
                # Technology improves productivity
                employee.productivity_index *= (1 + intensity * 0.08)
                
            elif action_type == 'leadership_development':
                # Leadership development affects leadership potential and engagement
                if employee.leadership_potential > 0.5:  # Only for high-potential employees
                    employee.leadership_potential = min(1.0, employee.leadership_potential + intensity * 0.1)
                    employee.engagement_level = min(1.0, employee.engagement_level + intensity * 0.05)
        
        # Apply some random decay to simulate natural changes
        self._apply_natural_changes()
    
    def _apply_natural_changes(self):
        """Apply natural changes to employee metrics over time"""
        for employee in self.employees:
            # Natural decay in some metrics
            employee.satisfaction_score *= np.random.uniform(0.98, 1.02)
            employee.engagement_level *= np.random.uniform(0.97, 1.01)
            employee.productivity_index *= np.random.uniform(0.99, 1.01)
            
            # Clip values to valid ranges
            employee.satisfaction_score = np.clip(employee.satisfaction_score, 0.1, 1.0)
            employee.engagement_level = np.clip(employee.engagement_level, 0.1, 1.0)
            employee.productivity_index = np.clip(employee.productivity_index, 0.3, 1.5)
            
            # Update tenure
            employee.tenure += 1
            
            # Simulate employee turnover
            if (employee.satisfaction_score < 0.4 or employee.engagement_level < 0.3) and np.random.random() < 0.05:
                # Employee leaves, replace with new hire
                self._replace_employee(employee)
    
    def _replace_employee(self, leaving_employee: EmployeeProfile):
        """Replace a leaving employee with a new hire"""
        new_employee = EmployeeProfile(
            employee_id=leaving_employee.employee_id,
            skills=np.random.uniform(0.4, 0.8, self.skill_dimensions),
            performance_history=[np.random.uniform(0.5, 0.8) for _ in range(3)],
            satisfaction_score=np.random.uniform(0.6, 0.8),
            engagement_level=np.random.uniform(0.5, 0.8),
            productivity_index=np.random.uniform(0.7, 1.1),
            learning_ability=np.random.uniform(0.6, 0.9),
            team_collaboration=np.random.uniform(0.5, 0.8),
            leadership_potential=np.random.uniform(0.2, 0.6),
            work_life_balance=np.random.uniform(0.6, 0.8),
            career_stage="Junior",
            department=leaving_employee.department,
            tenure=1
        )
        
        # Replace in the list
        idx = self.employees.index(leaving_employee)
        self.employees[idx] = new_employee
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on HR metrics performance vs targets"""
        current_metrics = self._calculate_hr_metrics()
        
        # Calculate performance relative to targets
        target_performance = {}
        metrics_dict = {
            'employee_satisfaction': current_metrics.employee_satisfaction,
            'employee_engagement': current_metrics.employee_engagement,
            'productivity_index': current_metrics.productivity_index,
            'retention_rate': current_metrics.retention_rate,
            'recruitment_quality': current_metrics.recruitment_quality,
            'training_effectiveness': current_metrics.training_effectiveness,
            'organizational_culture_score': current_metrics.organizational_culture_score,
            'leadership_effectiveness': current_metrics.leadership_effectiveness,
            'innovation_index': current_metrics.innovation_index
        }
        
        # Negative metrics (lower is better)
        negative_metrics = {
            'turnover_rate': current_metrics.turnover_rate,
            'absenteeism_rate': current_metrics.absenteeism_rate
        }
        
        # Calculate positive rewards
        positive_reward = 0
        for metric_name, metric_value in metrics_dict.items():
            target = self.performance_targets[metric_name]
            performance_ratio = metric_value / target
            positive_reward += max(0, performance_ratio - 1.0)  # Reward exceeding targets
        
        # Calculate negative penalties
        negative_penalty = 0
        for metric_name, metric_value in negative_metrics.items():
            target = self.performance_targets[metric_name]
            performance_ratio = metric_value / target
            negative_penalty += max(0, performance_ratio - 1.0)  # Penalty for exceeding thresholds
        
        # Combine rewards and penalties
        reward = positive_reward - negative_penalty
        
        # Add bonus for consistency (multiple metrics performing well)
        consistency_bonus = 0
        if len([1 for _, v in metrics_dict.items() if v / self.performance_targets[_] > 0.95]) >= 6:
            consistency_bonus = 0.5
        
        # Market condition adjustments
        market_multiplier = (
            self.market_conditions['economic_growth'] * 0.3 +
            self.market_conditions['industry_competitiveness'] * 0.2 +
            self.market_conditions['technology_advancement'] * 0.3 +
            self.market_conditions['regulatory_changes'] * 0.2
        )
        
        total_reward = (reward + consistency_bonus) * market_multiplier
        
        return total_reward
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.hr_metrics_history:
            return {}
        
        latest_metrics = self.hr_metrics_history[-1]
        
        summary = {
            'current_metrics': latest_metrics,
            'target_achievement': {},
            'employee_count': len(self.employees),
            'time_elapsed': self.current_time,
            'market_conditions': self.market_conditions
        }
        
        # Calculate target achievement rates
        metrics_dict = {
            'employee_satisfaction': latest_metrics.employee_satisfaction,
            'employee_engagement': latest_metrics.employee_engagement,
            'productivity_index': latest_metrics.productivity_index,
            'retention_rate': latest_metrics.retention_rate,
            'recruitment_quality': latest_metrics.recruitment_quality,
            'training_effectiveness': latest_metrics.training_effectiveness,
            'organizational_culture_score': latest_metrics.organizational_culture_score,
            'leadership_effectiveness': latest_metrics.leadership_effectiveness,
            'innovation_index': latest_metrics.innovation_index
        }
        
        for metric_name, metric_value in metrics_dict.items():
            target = self.performance_targets[metric_name]
            achievement_rate = (metric_value / target) * 100
            summary['target_achievement'][metric_name] = achievement_rate
        
        return summary