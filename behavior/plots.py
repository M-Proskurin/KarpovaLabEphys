import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


def plot_behavioral_summary(behavior, figsize=(16, 12), save_path=None):
    """
    Create comprehensive statistical summary plots for behavioral data.
    
    Parameters:
    behavior: Behavior class instance with trial data
    figsize: tuple, figure size (width, height)
    save_path: str, optional path to save the figure
    
    Returns:
    fig: matplotlib figure object
    """
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Extract data from behavior object
    trial_df = behavior.trial_summary
    n_trials = len(trial_df)
    
    # Calculate basic statistics
    reward_rate = np.mean(behavior.reward) * 100
    omission_rate = np.mean(behavior.omission) * 100
    abandonment_rate = np.mean(behavior.abandoned) * 100
    
    # Choice statistics
    left_choices = np.sum(behavior.choice == 1)
    right_choices = np.sum(behavior.choice == 0)
    choice_bias = (left_choices - right_choices) / (left_choices + right_choices) if (left_choices + right_choices) > 0 else 0
    
    # Create subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Trial outcomes pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    outcomes = ['Reward', 'Omission', 'Abandonment']
    values = [reward_rate, omission_rate, abandonment_rate]
    colors = ['green', 'orange', 'red']
    ax1.pie(values, labels=outcomes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Trial Outcomes')
    
    # 2. Choice distribution
    ax2 = fig.add_subplot(gs[0, 1])
    choice_counts = [left_choices, right_choices]
    choice_labels = ['Left', 'Right']
    bars = ax2.bar(choice_labels, choice_counts, color=['blue', 'red'], alpha=0.7)
    ax2.set_title(f'Choice Distribution\n(Bias: {choice_bias:.3f})')
    ax2.set_ylabel('Number of Trials')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Reward volume distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if not np.all(np.isnan(behavior.reward_volume)):
        ax3.hist(behavior.reward_volume[behavior.reward_volume > 0], 
                bins=20, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('Reward Volume Distribution')
        ax3.set_xlabel('Volume (μL)')
        ax3.set_ylabel('Frequency')
    else:
        ax3.text(0.5, 0.5, 'No reward volume data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Reward Volume Distribution')
    
    # 4. Session timeline - outcomes over trials
    ax4 = fig.add_subplot(gs[0, 3])
    trial_numbers = trial_df['trial'].values
    colors_timeline = []
    for i, row in trial_df.iterrows():
        if row['reward']:
            colors_timeline.append('green')
        elif row['omission']:
            colors_timeline.append('orange')
        elif row['abandoned']:
            colors_timeline.append('red')
        else:
            colors_timeline.append('gray')
    
    ax4.scatter(trial_numbers, np.ones(len(trial_numbers)), c=colors_timeline, alpha=0.6, s=10)
    ax4.set_title('Trial Timeline')
    ax4.set_xlabel('Trial Number')
    ax4.set_ylim(0.5, 1.5)
    ax4.set_yticks([])
    
    # 5. Running performance (reward rate over time)
    ax5 = fig.add_subplot(gs[1, :2])
    window_size = min(20, n_trials // 10)  # adaptive window size
    if window_size > 0:
        running_reward = pd.Series(behavior.reward).rolling(window=window_size, center=True).mean() * 100
        ax5.plot(trial_numbers, running_reward, linewidth=2, color='green', alpha=0.8)
        ax5.axhline(y=reward_rate, color='green', linestyle='--', alpha=0.5, label=f'Overall: {reward_rate:.1f}%')
        ax5.set_title(f'Running Reward Rate (window={window_size})')
        ax5.set_xlabel('Trial Number')
        ax5.set_ylabel('Reward Rate (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Choice bias over time
    ax6 = fig.add_subplot(gs[1, 2:])
    if window_size > 0:
        # Calculate running choice bias
        running_choice = []
        for i in range(len(behavior.choice)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(behavior.choice), i + window_size//2 + 1)
            window_choices = behavior.choice[start_idx:end_idx]
            valid_choices = window_choices[window_choices >= 0]  # exclude invalid choices
            if len(valid_choices) > 0:
                left_prop = np.mean(valid_choices)  # proportion of left choices (1s)
                bias = 2 * left_prop - 1  # convert to bias (-1 to 1)
                running_choice.append(bias)
            else:
                running_choice.append(0)
        
        ax6.plot(trial_numbers, running_choice, linewidth=2, color='purple', alpha=0.8)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.axhline(y=choice_bias, color='purple', linestyle='--', alpha=0.5, label=f'Overall: {choice_bias:.3f}')
        ax6.set_title(f'Running Choice Bias (window={window_size})')
        ax6.set_xlabel('Trial Number')
        ax6.set_ylabel('Choice Bias (L-R)/(L+R)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Reaction times (if available)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'c_in' in trial_df.columns and 'side_in' in trial_df.columns:
        # Calculate reaction time as difference between center out and side in
        if 'c_out' in trial_df.columns:
            reaction_times = trial_df['side_in'] - trial_df['c_out']
            reaction_times = reaction_times.dropna()
            if len(reaction_times) > 0:
                ax7.hist(reaction_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax7.set_title(f'Reaction Times\n(μ={np.mean(reaction_times):.3f}s)')
                ax7.set_xlabel('Time (s)')
                ax7.set_ylabel('Frequency')
            else:
                ax7.text(0.5, 0.5, 'No reaction time data', ha='center', va='center', transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.5, 'Incomplete timing data', ha='center', va='center', transform=ax7.transAxes)
    else:
        ax7.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax7.transAxes)
    ax7.set_title('Reaction Times')
    
    # 8. Outcome by choice
    ax8 = fig.add_subplot(gs[2, 1])
    choice_outcome_data = []
    for choice_val, choice_name in [(0, 'Right'), (1, 'Left')]:
        choice_mask = behavior.choice == choice_val
        if np.any(choice_mask):
            reward_rate_choice = np.mean(behavior.reward[choice_mask]) * 100
            choice_outcome_data.append((choice_name, reward_rate_choice))
    
    if choice_outcome_data:
        choices, rates = zip(*choice_outcome_data)
        bars = ax8.bar(choices, rates, color=['red', 'blue'], alpha=0.7)
        ax8.set_title('Reward Rate by Choice')
        ax8.set_ylabel('Reward Rate (%)')
        for bar, rate in zip(bars, rates):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    # 9. Streak analysis
    ax9 = fig.add_subplot(gs[2, 2])
    if hasattr(behavior.extractor, 'extract_choices_and_streaks'):
        try:
            streak_df = behavior.extractor.extract_choices_and_streaks()
            if 'streak_length' in streak_df.columns:
                streak_lengths = streak_df['streak_length'].dropna()
                if len(streak_lengths) > 0:
                    ax9.hist(streak_lengths, bins=range(1, int(streak_lengths.max()) + 2), 
                            alpha=0.7, color='gold', edgecolor='black')
                    ax9.set_title(f'Choice Streak Lengths\n(max={int(streak_lengths.max())})')
                    ax9.set_xlabel('Streak Length')
                    ax9.set_ylabel('Frequency')
                else:
                    ax9.text(0.5, 0.5, 'No streak data', ha='center', va='center', transform=ax9.transAxes)
            else:
                ax9.text(0.5, 0.5, 'No streak column', ha='center', va='center', transform=ax9.transAxes)
        except:
            ax9.text(0.5, 0.5, 'Error extracting streaks', ha='center', va='center', transform=ax9.transAxes)
    else:
        ax9.text(0.5, 0.5, 'No streak method', ha='center', va='center', transform=ax9.transAxes)
    ax9.set_title('Choice Streak Lengths')
    
    # 10. Summary statistics text
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')
    
    # Calculate additional statistics
    if len(trial_df) > 0:
        session_duration = behavior.extractor.session_duration if hasattr(behavior.extractor, 'session_duration') else 'N/A'
        max_vol = behavior.extractor.max_vol if hasattr(behavior.extractor, 'max_vol') else 'N/A'
        
        stats_text = f"""Session Summary:
        
Subject: {behavior.rat}
Date: {behavior.date}
Duration: {session_duration}

Total Trials: {n_trials}
Reward Rate: {reward_rate:.1f}%
Omission Rate: {omission_rate:.1f}%
Abandonment: {abandonment_rate:.1f}%

Choice Bias: {choice_bias:.3f}
Left Choices: {left_choices}
Right Choices: {right_choices}

Max Volume: {max_vol} mL"""
    else:
        stats_text = "No trial data available"
    
    ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add main title
    fig.suptitle(f'Behavioral Summary - {behavior.rat} ({behavior.date})', 
                fontsize=16, fontweight='bold')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_trial_by_trial_analysis(behavior, trial_range=None, figsize=(14, 8)):
    """
    Create detailed trial-by-trial analysis plots.
    
    Parameters:
    behavior: Behavior class instance
    trial_range: tuple (start, end) or None for all trials
    figsize: tuple, figure size
    
    Returns:
    fig: matplotlib figure object
    """
    
    trial_df = behavior.trial_summary
    
    if trial_range:
        start, end = trial_range
        trial_df = trial_df[(trial_df['trial'] >= start) & (trial_df['trial'] <= end)]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Trial outcomes over time
    ax1 = axes[0, 0]
    trials = trial_df['trial'].values
    
    # Create color map for outcomes
    colors = []
    for _, row in trial_df.iterrows():
        if row['reward']:
            colors.append('green')
        elif row['omission']:
            colors.append('orange')
        elif row['abandoned']:
            colors.append('red')
        else:
            colors.append('gray')
    
    ax1.scatter(trials, np.ones(len(trials)), c=colors, alpha=0.7, s=50)
    ax1.set_title('Trial Outcomes')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylim(0.5, 1.5)
    ax1.set_yticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Reward'),
                      Patch(facecolor='orange', label='Omission'),
                      Patch(facecolor='red', label='Abandonment')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Choices over time
    ax2 = axes[0, 1]
    choice_data = behavior.choice[behavior.choice >= 0]  # exclude invalid choices
    choice_trials = trial_df['trial'].values[behavior.choice >= 0]
    
    choice_colors = ['red' if c == 0 else 'blue' for c in choice_data]
    ax2.scatter(choice_trials, choice_data, c=choice_colors, alpha=0.7, s=50)
    ax2.set_title('Choices Over Time')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Choice')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Right', 'Left'])
    
    # 3. Reward volume over time (if available)
    ax3 = axes[1, 0]
    if not np.all(np.isnan(behavior.reward_volume)):
        reward_trials = trial_df[trial_df['reward']]['trial'].values
        reward_volumes = behavior.reward_volume[behavior.reward == 1]
        ax3.plot(reward_trials, reward_volumes, 'o-', color='green', alpha=0.7)
        ax3.set_title('Reward Volume Over Time')
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('Volume (μL)')
    else:
        ax3.text(0.5, 0.5, 'No reward volume data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Reward Volume Over Time')
    
    # 4. Timing analysis (if available)
    ax4 = axes[1, 1]
    if all(col in trial_df.columns for col in ['c_in', 'c_out', 'side_in']):
        # Calculate decision time (center in to center out)
        decision_times = trial_df['c_out'] - trial_df['c_in']
        decision_times = decision_times.dropna()
        
        if len(decision_times) > 0:
            ax4.plot(trial_df['trial'].values[:len(decision_times)], decision_times, 'o-', alpha=0.7)
            ax4.set_title('Decision Times')
            ax4.set_xlabel('Trial Number')
            ax4.set_ylabel('Time (s)')
        else:
            ax4.text(0.5, 0.5, 'No valid timing data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Timing data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Decision Times')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage:
    # from behavior.parse_pycontrol import Behavior
    # plot_behavioral_summary(behavior_instance)
    # plot_trial_by_trial_analysis(behavior_instance, trial_range=(1, 100))
    pass