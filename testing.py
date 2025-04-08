import PySimpleGUI as sg

layout = [
    [sg.Button('Show Popup')]
]

window = sg.Window('Popup Example', layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == 'Show Popup':
        response = sg.popup_yes_no('Do you want to proceed?', title='Popup with Two Buttons')
        
        if response == 'Yes':
            print('User clicked Yes')
        else:
            print('User clicked No')

window.close()
