from flask_wtf import FlaskForm
from wtforms import *
from wtforms.validators import DataRequired
from random import *


class MultiCheckboxField(SelectMultipleField):
    widget = widgets.ListWidget(prefix_label=False)
    option_widget = widgets.CheckboxInput()



class DFAForm(FlaskForm):

	states = StringField("States (e.g 'A,B,C')", validators=[DataRequired()])
	alphabet = StringField("Alphabet (e.g '0,1,2')", validators=[DataRequired()])
	start_state = SelectField("Start State", choices = [("", "")])
	accept_states = MultiCheckboxField('Accepting States')
	create = SubmitField("Create DFA")
	minimize = SubmitField("Minimize")
	string_input = StringField("Input")
	simulate = SubmitField("Simulate")
	step = SubmitField("Step")
	remaining = StringField("Remaining")
	path = StringField("Path")
	result = StringField("Result")
	reset = SubmitField("Reset")


class NFAForm(FlaskForm):

	states = StringField("States (e.g 'A,B,C')", validators=[DataRequired()])
	alphabet = StringField("Alphabet (e.g '0,1,2')", validators=[DataRequired()])
	start_state = SelectField("Start State", choices = [("", "")])
	accept_states = MultiCheckboxField('Accepting States')
	create = SubmitField("Create NFA")
	string_input = StringField("Input")
	simulate = SubmitField("Simulate")
	paths = TextAreaField("Paths")
	result = StringField("Result")
	reset = SubmitField("Reset")
	convert = SubmitField("Convert to DFA")





