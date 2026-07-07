

class TextBoxInput:
    
    """
    TextBoxInput Contract

    What:
        A utility module that handles ext input in the text box.

    Why:
        To enable real-time text manipulation from keyboard events in interactive
        computer vision or GUI-driven applications.
        
    Inputs:
        key (int)
            Integer key code from keyboard input event system.

        text (str)
            Current text buffer being modified by user input.

    Process:
        Receive keyboard key input
        Check for special key events (Backspace, Tab, Escape)
        If Backspace is pressed, remove last character from text
        If valid printable ASCII key is pressed, convert key to character and append
        Ignore Tab and Escape keys for text modification
        Return updated text buffer

    Output:
        str
            Updated text string after applying keyboard input rules
    """
    
    #Use to handle keyboard input.
    def update_text_input(self, key: int, text: str) -> str:

        #CONSTANTS.
        TAB_KEY = 9
        ESC_KEY = 27

        #When Backspace is entered it subtract the text string.
        if key == 8: 
            text = text[:-1]

        #All keys except the conditions will update the text variable.
        #Tab key and ESC key should NOT MODITY the text.
        elif 32 <= key <= 126 and key != TAB_KEY and key != ESC_KEY:
            text += chr(key)

        return text
