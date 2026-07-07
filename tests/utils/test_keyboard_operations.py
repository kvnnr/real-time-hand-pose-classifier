from src.feature.textbox.text_box_input import TextBoxInput

class TestUpdateTextInput:

    """
    Unit tests for KeyboardOps.update_text_input().

    Tests:
    - Backspace removes the last character.
    - Reserved keys are ignored.
    - Printable characters are appended.s
    """

    """
    Variables:
        kb_op -> object

        self.text
        self.subtracted_text
        self.tab_key
        self.esc_key
        self.empty_text
        self.backspace_key
        self.letter_t_key
    """
    def setup_method(self):

        #Object creation,
        self.kb_op = TextBoxInput()
        
        #Sample Inputs.
        self.text = 'Test'
        self.subtracted_text = "Tes"
        self.tab_key = 9
        self.esc_key = 27
        self.empty_text = ""
        self.backspace_key = 8
        self.letter_t_key = 116 

    #Check the backspace key in text box.
    def test_remove_text_on_backspace_key(self):
        
        #Test if backspace works in the text operation.
        text = self.kb_op.update_text_input(self.backspace_key, self.text)
        assert text == self.subtracted_text, \
            "Error: Backspace in text box is not working."

    # ESC and TAB should NOT modify text (ignored inputs) 
    def test_ignore_esc_n_tab_key(self):
        
        #Test if it ignore the esc key.
        text_1 = self.kb_op.update_text_input(self.esc_key, self.text)
        assert text_1 == self.text, \
            "Error: ESC should not modify text."
        
        #Test if it ignore the tab key.
        text_2 = self.kb_op.update_text_input(self.tab_key, self.text)
        assert text_2 == self.text, \
            "Error: TAB should not modify text."
    
    #Check for happy path.
    def test_accept_valid_input(self):

        #Test accept the valid inputs.
        text = self.kb_op.update_text_input(self.letter_t_key, self.empty_text)
        assert text == "t", \
            "Error: Valid key input was not appended correctly."
    
    # Edge case: backspace on empty string should not crash
    def test_backspace_on_empty_string(self):

        text = self.kb_op.update_text_input(self.backspace_key,self.empty_text)
        assert text == "", \
            "Error: Backspace on empty string should return empty string."