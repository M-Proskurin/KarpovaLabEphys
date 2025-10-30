import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Bokeh imports for interactive plotting
import bokeh.plotting as bk
from bokeh.models import (
    ColumnDataSource,
    CDSView,
    BooleanFilter as BF,
    Legend,
    HoverTool,
    Div,
    CustomJS,
    LabelSet,
    Spinner,
    Button,
    CrosshairTool,
    PreText,
    DatetimeTickFormatter,
    DataTable,
    TableColumn,
    TabPanel,
    Tabs,
    RangeTool,
    BoxAnnotation,
    LegendItem,
    CheckboxGroup,
)
from bokeh.layouts import column, row
from bokeh import events

# Color palette similar to sequence_plotter
ORANGE = "#FF8C00"
BLUE = "#1E90FF"
GREEN = "#32CD32"
RED = "#DC143C"
GREY = "#708090"
WHITE = "#FFFFFF"
YELLOW = "#FFD700"
GREEN_LIGHT = "#90EE90"
ORANGE_LIGHT = "#FFB347"
BLUE_LIGHT = "#87CEEB"

PLOT_HEIGHT = 280
MARKER_SIZE = 10.5
SMALL_MARKER_SIZE = 3
Y_RANGE_PLUS_MINUS = 7


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
    ypos = []
    for i, row in trial_df.iterrows():
        if row['reward']:
            colors_timeline.append('green')
            ypos.append(2)
        elif row['omission']:
            colors_timeline.append('orange')
            ypos.append(0)
        elif row['abandoned']:
            colors_timeline.append('red')
            ypos.append(1)
        else:
            colors_timeline.append('gray')
            ypos.append(0)

    ax4.scatter(trial_numbers, ypos, c=colors_timeline, alpha=0.6, s=10)
    ax4.set_title('Trial Timeline')
    ax4.set_xlabel('Trial Number')
    ax4.set_ylim(-0.5, 2.5)
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
            reaction_times = trial_df['side_in'] - trial_df['c_in']
            reaction_times = reaction_times.dropna()
            if len(reaction_times) > 0:
                ax7.hist(reaction_times, np.arange(0,5,0.1), alpha=0.7, color='skyblue', edgecolor='black')
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
            if 'streak_counts' in streak_df.columns:
                streak_lengths = streak_df['streak_counts'].dropna()
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
            ax4.set_yscale('log')  # Log scale for better visualization
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


def plot_interactive_behavior(behavior, save_path=None, show=True):
    """
    Create an interactive behavioral plot using Bokeh with scrollable timeline.
    Similar to sequence_plotter.py style.
    
    Parameters:
    behavior: Behavior class instance with trial data
    save_path: str, optional path to save HTML file
    show: bool, whether to show the plot in browser
    
    Returns:
    bokeh layout object
    """
    
    # Prepare data for Bokeh
    trial_df = behavior.trial_summary
    n_trials = len(trial_df)
    
    # Create time data (use trial timing if available, otherwise simulate)
    if hasattr(behavior, 'times') and behavior.times is not None:
        # Use actual timing data
        times_ms = behavior.times[:, 0] * 1000  # Convert to milliseconds for better precision
    else:
        # Simulate timing data - assume 30 seconds per trial on average
        times_ms = np.cumsum(np.random.exponential(30, n_trials)) * 1000
    
    # Prepare choice data for plotting
    choice_y = []
    choice_colors = []
    for choice in behavior.choice:
        if choice == 1:  # Left
            choice_y.append(1)
            choice_colors.append(ORANGE)
        elif choice == 0:  # Right
            choice_y.append(-1)
            choice_colors.append(BLUE)
        else:  # Invalid/No choice
            choice_y.append(0)
            choice_colors.append(GREY)
    
    # Prepare outcome colors
    outcome_colors = []
    outcome_sizes = []
    for i, (reward, omission, abandoned) in enumerate(zip(behavior.reward, behavior.omission, behavior.abandoned)):
        if reward:
            outcome_colors.append(GREEN)
            outcome_sizes.append(MARKER_SIZE + 2)
        elif omission:
            outcome_colors.append(YELLOW)
            outcome_sizes.append(MARKER_SIZE)
        elif abandoned:
            outcome_colors.append(RED)
            outcome_sizes.append(MARKER_SIZE - 2)
        else:
            outcome_colors.append(WHITE)
            outcome_sizes.append(MARKER_SIZE - 3)
    
    # Calculate running statistics
    window_size = min(20, n_trials // 10)
    running_reward = pd.Series(behavior.reward).rolling(window=window_size, center=True).mean() * 100
    running_bias = []
    for i in range(len(behavior.choice)):
        start_idx = max(0, i - window_size//2)
        end_idx = min(len(behavior.choice), i + window_size//2 + 1)
        window_choices = behavior.choice[start_idx:end_idx]
        valid_choices = window_choices[window_choices >= 0]
        if len(valid_choices) > 0:
            left_prop = np.mean(valid_choices)
            bias = 2 * left_prop - 1
            running_bias.append(bias)
        else:
            running_bias.append(0)
    
    # Create data source
    source_data = {
        'trials': trial_df['trial'].values,
        'trial_str': trial_df['trial'].astype(str).values,
        'times': times_ms[:len(trial_df)],
        'choice_y': choice_y,
        'choice_colors': choice_colors,
        'outcome_colors': outcome_colors,
        'outcome_sizes': outcome_sizes,
        'reward': behavior.reward,
        'omission': behavior.omission,
        'abandoned': behavior.abandoned,
        'reward_volume': behavior.reward_volume,
        'running_reward': running_reward.fillna(0).values,
        'running_bias': running_bias,
        'zeros': np.zeros(len(trial_df)),
    }
    
    # Add timing data if available
    if 'c_in' in trial_df.columns:
        source_data['c_in'] = trial_df['c_in'].fillna(0).values
        source_data['c_out'] = trial_df['c_out'].fillna(0).values
        source_data['side_in'] = trial_df['side_in'].fillna(0).values
        source_data['side_out'] = trial_df['side_out'].fillna(0).values
        # Calculate reaction times
        reaction_times = (trial_df['side_in'] - trial_df['c_out']).fillna(0).values
        source_data['reaction_time'] = reaction_times
    else:
        # Fill with zeros if no timing data
        for col in ['c_in', 'c_out', 'side_in', 'side_out', 'reaction_time']:
            source_data[col] = np.zeros(len(trial_df))
    
    source = ColumnDataSource(data=source_data)
    
    # Create plots
    # Main trial plot
    trial_plot = bk.figure(
        title="Trial Outcomes and Choices",
        width=900,
        height=PLOT_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        active_scroll="wheel_zoom",
        x_axis_label="Trial Number",
        y_range=(-Y_RANGE_PLUS_MINUS, Y_RANGE_PLUS_MINUS),
        toolbar_location="above"
    )
    
    # Time plot
    time_plot = bk.figure(
        title="Behavior Over Time",
        width=900,
        height=PLOT_HEIGHT,
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        active_scroll="wheel_zoom",
        x_axis_label="Time (ms)",
        x_axis_type="datetime",
        y_range=(-Y_RANGE_PLUS_MINUS, Y_RANGE_PLUS_MINUS),
        toolbar_location="above"
    )
    
    # Performance plot
    perf_plot = bk.figure(
        title="Running Performance",
        width=900,
        height=200,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_axis_label="Trial Number",
        y_axis_label="Performance (%)",
        toolbar_location="above"
    )
    
    # Bias plot
    bias_plot = bk.figure(
        title="Choice Bias Over Time",
        width=900,
        height=200,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_axis_label="Trial Number",
        y_axis_label="Choice Bias",
        y_range=(-1.1, 1.1),
        toolbar_location="above"
    )
    
    # Add choice markers
    trial_choices = trial_plot.scatter(
        'trials', 'choice_y', 
        source=source,
        size=10,
        color='choice_colors',
        alpha=0.7,
        legend_label="Choices"
    )
    
    time_choices = time_plot.scatter(
        'times', 'choice_y',
        source=source,
        size=10,
        color='choice_colors',
        alpha=0.7
    )
    
    # Add outcome markers
    # Create views for different outcomes
    reward_view = CDSView(filter=BF([bool(r) for r in behavior.reward]))
    omission_view = CDSView(filter=BF([bool(o) for o in behavior.omission]))
    abandoned_view = CDSView(filter=BF([bool(a) for a in behavior.abandoned]))
    
    # Trial plot outcomes
    trial_rewards = trial_plot.scatter(
        'trials', 'choice_y',
        source=source,
        view=reward_view,
        size='outcome_sizes',
        color=GREEN,
        marker='circle',
        alpha=0.8,
        legend_label="Rewards"
    )
    
    trial_omissions = trial_plot.scatter(
        'trials', 'choice_y',
        source=source,
        view=omission_view,
        size='outcome_sizes',
        color=YELLOW,
        marker='triangle',
        alpha=0.8,
        legend_label="Omissions"
    )
    
    trial_abandoned = trial_plot.scatter(
        'trials', 'choice_y',
        source=source,
        view=abandoned_view,
        size='outcome_sizes',
        color=RED,
        marker='square',
        alpha=0.8,
        legend_label="Abandoned"
    )
    
    # Time plot outcomes
    time_rewards = time_plot.scatter(
        'times', 'choice_y',
        source=source,
        view=reward_view,
        size='outcome_sizes',
        color=GREEN,
        marker='circle',
        alpha=0.8
    )
    
    time_omissions = time_plot.scatter(
        'times', 'choice_y',
        source=source,
        view=omission_view,
        size='outcome_sizes',
        color=YELLOW,
        marker='triangle',
        alpha=0.8
    )
    
    time_abandoned = time_plot.scatter(
        'times', 'choice_y',
        source=source,
        view=abandoned_view,
        size='outcome_sizes',
        color=RED,
        marker='square',
        alpha=0.8
    )
    
    # Add performance lines
    perf_plot.line('trials', 'running_reward', source=source, 
                   line_width=2, color=GREEN, legend_label="Reward Rate")
    
    bias_plot.line('trials', 'running_bias', source=source,
                   line_width=2, color=BLUE, legend_label="Choice Bias")
    bias_plot.line('trials', 'zeros', source=source,
                   line_width=1, color=GREY, line_dash='dashed')
    
    # Add hover tools
    trial_hover = HoverTool(
        renderers=[trial_choices, trial_rewards, trial_omissions, trial_abandoned],
        tooltips=[
            ("Trial", "@trials"),
            ("Choice", "@choice_y{0.0}"),
            ("Reward", "@reward"),
            ("Volume", "@reward_volume μL"),
            ("Reaction Time", "@reaction_time{0.000}s"),
        ]
    )
    trial_plot.add_tools(trial_hover)
    
    time_hover = HoverTool(
        renderers=[time_choices, time_rewards, time_omissions, time_abandoned],
        tooltips=[
            ("Trial", "@trials"),
            ("Time", "@times{0.0}ms"),
            ("Choice", "@choice_y{0.0}"),
            ("Reward", "@reward"),
            ("Volume", "@reward_volume μL"),
        ]
    )
    time_plot.add_tools(time_hover)
    
    # Customize plots
    for plot in [trial_plot, time_plot]:
        plot.yaxis.ticker = [-2, -1, 0, 1, 2]
        plot.yaxis.major_label_overrides = {
            -2: "R2", -1: "R1", 0: "Center", 1: "L1", 2: "L2"
        }
        plot.legend.click_policy = "hide"
        plot.legend.location = "top_left"
    
    # Add crosshairs
    trial_plot.add_tools(CrosshairTool())
    time_plot.add_tools(CrosshairTool())
    
    # Create range selector plots
    trial_selector = bk.figure(
        height=100,
        width=900,
        tools="",
        toolbar_location=None,
        y_range=(-2, 2)
    )
    
    time_selector = bk.figure(
        height=100,
        width=900,
        tools="",
        toolbar_location=None,
        x_axis_type="datetime",
        y_range=(-2, 2)
    )
    
    # Add data to selectors
    trial_selector.scatter('trials', 'choice_y', source=source, 
                          size=SMALL_MARKER_SIZE, color='choice_colors', alpha=0.5)
    time_selector.scatter('times', 'choice_y', source=source,
                         size=SMALL_MARKER_SIZE, color='choice_colors', alpha=0.5)
    
    # Add range tools
    trial_range_tool = RangeTool(x_range=trial_plot.x_range)
    trial_range_tool.overlay.fill_color = "yellow"
    trial_range_tool.overlay.fill_alpha = 0.2
    trial_selector.add_tools(trial_range_tool)
    
    time_range_tool = RangeTool(x_range=time_plot.x_range)
    time_range_tool.overlay.fill_color = "yellow"
    time_range_tool.overlay.fill_alpha = 0.2
    time_selector.add_tools(time_range_tool)
    
    # Create statistics div
    stats_div = create_stats_div(behavior)
    
    # Create controls
    goto_trial_spinner = Spinner(low=1, high=n_trials, step=1, value=1, width=100)
    goto_trial_btn = Button(label="Go to Trial", button_type="primary", width=100)
    
    # JavaScript callback for goto trial
    goto_callback = CustomJS(
        args=dict(
            spinner=goto_trial_spinner,
            trial_range=trial_plot.x_range,
            time_range=time_plot.x_range,
            source=source
        ),
        code="""
        const trial = spinner.value;
        const trial_data = source.data['trials'];
        const time_data = source.data['times'];
        
        // Find the index of the trial
        const trial_idx = trial_data.indexOf(trial);
        
        if (trial_idx >= 0) {
            // Center trial plot on selected trial
            const window_size = (trial_range.end - trial_range.start) / 2;
            trial_range.start = trial - window_size;
            trial_range.end = trial + window_size;
            
            // Center time plot on corresponding time
            const trial_time = time_data[trial_idx];
            const time_window = (time_range.end - time_range.start) / 2;
            time_range.start = trial_time - time_window;
            time_range.end = trial_time + time_window;
        }
        """
    )
    goto_trial_btn.js_on_event(events.ButtonClick, goto_callback)
    
    # Layout
    controls = row(
        goto_trial_btn, goto_trial_spinner,
        sizing_mode="fixed"
    )
    
    layout = column(
        stats_div,
        controls,
        trial_plot,
        trial_selector,
        time_plot,
        time_selector,
        perf_plot,
        bias_plot,
        sizing_mode="stretch_width"
    )
    
    # Create tabs
    main_tab = TabPanel(child=layout, title="Behavioral Analysis")
    tabs = Tabs(tabs=[main_tab])
    
    # Show or save
    if save_path:
        bk.output_file(save_path)
        bk.save(tabs)
        print(f"Interactive plot saved to: {save_path}")
    
    if show:
        bk.show(tabs)
    
    return tabs


def create_stats_div(behavior):
    """Create a statistics summary div for the interactive plot."""
    
    # Calculate basic statistics
    n_trials = len(behavior.trial_summary)
    reward_rate = np.mean(behavior.reward) * 100
    omission_rate = np.mean(behavior.omission) * 100
    abandonment_rate = np.mean(behavior.abandoned) * 100
    
    left_choices = np.sum(behavior.choice == 1)
    right_choices = np.sum(behavior.choice == 0)
    choice_bias = (left_choices - right_choices) / (left_choices + right_choices) if (left_choices + right_choices) > 0 else 0
    
    session_duration = behavior.extractor.session_duration if hasattr(behavior.extractor, 'session_duration') else 'N/A'
    
    stats_html = f"""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #333;">
            Subject: <span style="color: {ORANGE};">{behavior.rat}</span> | 
            Date: <span style="color: {BLUE};">{behavior.date}</span> | 
            Duration: {session_duration}
        </h3>
        <div style="display: flex; justify-content: space-around; margin-top: 10px;">
            <div><strong>Trials:</strong> {n_trials}</div>
            <div><strong>Reward Rate:</strong> <span style="color: {GREEN};">{reward_rate:.1f}%</span></div>
            <div><strong>Omission Rate:</strong> <span style="color: orange;">{omission_rate:.1f}%</span></div>
            <div><strong>Abandonment:</strong> <span style="color: {RED};">{abandonment_rate:.1f}%</span></div>
            <div><strong>Choice Bias:</strong> {choice_bias:.3f}</div>
        </div>
    </div>
    """
    
    return Div(text=stats_html, sizing_mode="stretch_width")


if __name__ == "__main__":
    # Example usage:
    # from behavior.parse_pycontrol import Behavior
    # plot_behavioral_summary(behavior_instance)
    # plot_trial_by_trial_analysis(behavior_instance, trial_range=(1, 100))
    pass