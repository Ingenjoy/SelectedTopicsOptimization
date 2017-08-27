"""
Created on Thursday 22 August 2017
Last update: Friday 25 August 2017

@author: Michiel Stock
michielfmstock@gmail.com

Contains the color scheme of the course
"""

# COLORS
# ------

blue = '#264653'
green = '#2a9d8f'
yellow = '#e9c46a'
orange = '#f4a261'
red = '#e76f51'
black = '#50514F'

colors_list = [blue, red, green, orange, yellow, black]
#colors_list.reverse()

colors_dict = {'blue' : blue,
        'green' : green,
        'yellow' : yellow,
        'orange' : orange,
        'red' : red,
        'black' : black}


if __name__ == '__main__':
    import seaborn as sns
    sns.palplot(colors_list)
