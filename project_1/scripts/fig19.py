import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.font_manager as fm

#loading data
data = pd.read_csv("../data/academic_schedule_25_26.csv")

# preparing data
data['Start Date'] = pd.to_datetime(data['Start Date'], format='%d.%m.%Y')
data['End Date'] = pd.to_datetime(data['End Date'], format='%d.%m.%Y')
data['Duration'] = (data['End Date'] - data['Start Date']).dt.days + 1
semester_textures = {"Winter": "/", "Summer": "|", 'Vacation': "*"}

abbreviations = {
    'Decision period for academic year completion (specific study programs)': 'Decision period',
    'Written certification exams (foreign languages)': 'Language exams',
    'Course linking period for the next academic year (winter semester and full year)': 'Linkage period (NY)',
    'Final deadline to withdraw from summer semester course credit': 'Withdrawal deadline (S)',
    'Final deadline to withdraw from winter semester course credit': 'Withdrawal deadline (W)',
    'Deadline to remove course link to study program for summer semester': 'Linkage deadline (S)',
    'Deadline to remove course link to study program for winter semester':'Linkage deadline (W)',
    'Decision period for winter semester completion (specific study programs)': 'Decision period',
    'Retake written certification exams (English B2 level)': 'Retake English exams',
    'Course linking period for summer semester':'Linkage period (S)',
    'Retake Exam Session (Winter)': 'Retake Exams (W)',
    'Retake Exam Session (Summer)': 'Retake Exams (S)'
}


# sort by start date and duration
data = data.sort_values(['Start Date', 'Duration'], ascending=[True, False])

min_date = data['Start Date'].min() - timedelta(days=5)
max_date = data['End Date'].max() + timedelta(days=5)
months = pd.date_range(start=min_date, end=max_date, freq='MS')
num_events = len(data)
bar_height = 0.8
y_positions = np.arange(num_events)


#plot
fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=100)

# title
plt.title('Academic Schedule 2025-2026', fontsize=35, fontweight='bold')
plt.xlabel('Date', fontsize=25)
plt.ylabel('Event', fontsize=25)
plt.xticks(rotation=45)

# axis +grid
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0)) 
ax.set_yticks(y_positions)
ax.set_yticklabels([])
ax.set_xlim(mdates.date2num(min_date), mdates.date2num(max_date))
ax.set_ylim(-0.5, num_events - 0.5)
ax.grid(axis='x', which='major', linestyle='-', alpha=0.7)
ax.grid(axis='x', which='minor', linestyle=':', alpha=0.4)

for i, (_, event) in enumerate(data.iterrows()):
    start = event['Start Date']
    end = event['End Date']
    duration = (end - start).days + 1
    label = abbreviations.get(event['Event']) if event['Event'] in abbreviations else event['Event']
    
    ax.barh(y_positions[i], duration, height=bar_height, left=mdates.date2num(start), 
            color="lightgray", edgecolor='black', alpha=0.8, hatch = semester_textures.get(event['Semester']))
    
    #labels
    bar_width = mdates.date2num(end) - mdates.date2num(start)
    ax.text(mdates.date2num(end) + 2, y_positions[i], 
            label, ha='left', va='center', 
            fontweight='bold', fontsize=9)
    texture = semester_textures.get(event['Semester'], "")

# legend + abbreviation explanations
legend_elements = [Rectangle((0, 0), 1, 1,  facecolor = 'lightgray', label=semester, hatch = hatch) for semester,hatch in semester_textures.items()]
ax.legend(handles=legend_elements, loc = 'lower right',frameon=True, fontsize=12, title ="Period")
plt.figtext(0.07, 0.85, "Abbreavations:", fontsize=10, wrap=True, fontweight = 'bold')
abbrev_text = "\n".join([f"{abbr}: {full}" for full, abbr in abbreviations.items()])
plt.figtext(0.07, 0.63, abbrev_text, fontsize=9, wrap=True)


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('../images/fig19.pdf', bbox_inches='tight')
plt.savefig('../images/fig19.png', bbox_inches='tight', dpi=300)
plt.close()