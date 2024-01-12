import easygui

# List of options
genders = ["Male", "Female"]
regions = ["North East", "South", "Midwest", "West"]
age_groups = ["0-18", "19-27", "28-36", "37-45", "46-55", "55+"]
seasons = ["Winter", "Spring", "Summer", "Fall"]

# Dictionary to map options to integer values
gender_mapping = {'Male': 0, 'Female': 1}
region_mapping = {'North East': 0, 'Midwest': 1, 'South': 2, 'West': 3}
age_group_mapping = {'0-18': 0, '19-27': 1, '28-36': 2, '37-45': 3, '46-55': 4, '55+': 5}
season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}

# Function to get user selection with checkboxes
def get_user_selection():
    selected_genders = None
    while selected_genders is None:
        selected_genders = easygui.choicebox("Select Gender", choices=genders)
        if selected_genders is None:
            easygui.msgbox("Gender is required. Please try again.", "Error")
        
    selected_regions = None
    while selected_regions is None:
        selected_regions = easygui.choicebox("Select Region", choices=regions)
        if selected_regions is None:
            easygui.msgbox("Region is required. Please try again.", "Error")

    selected_age_groups = None
    while selected_age_groups is None:
        selected_age_groups = easygui.choicebox("Select Age Group", choices=age_groups)
        if selected_age_groups is None:
            easygui.msgbox("Age group is required. Please try again.", "Error")

    selected_seasons = None
    while selected_seasons is None:
        selected_seasons = easygui.choicebox("Select Season", choices=seasons)
        if selected_seasons is None:
            easygui.msgbox("Season is required. Please try again.", "Error")


    return {
        'Gender': selected_genders,
        'Region': selected_regions,
        'Age Group': selected_age_groups,
        'Season': selected_seasons
    }
    
    

# Initial user selection
user_selection = get_user_selection()