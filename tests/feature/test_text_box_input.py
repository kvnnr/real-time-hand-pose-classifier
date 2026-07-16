import pytest

from src.feature.textbox.text_box_input import TextBoxInput


class TestUpdateTextInput:

    def setup_method(self):

        """
        Variables:
            self.box: TextBoxInput instance
        """

        self.box = TextBoxInput()

    #Test backspace removes the last character case.
    def test_update_text_input_backspace_removes_last_character(self):

        result = self.box.update_text_input(8, "fist")

        assert result == "fis"

    #Test backspace on empty text returns empty text case.
    def test_update_text_input_backspace_on_empty_text_returns_empty(self):

        result = self.box.update_text_input(8, "")

        assert result == ""

    #Test printable ASCII key appends its character case.
    def test_update_text_input_printable_key_appends_character(self):

        result = self.box.update_text_input(ord("a"), "fis")

        assert result == "fisa"

    #Test lower boundary printable key (space) appends its character case.
    def test_update_text_input_space_key_appends_character(self):

        result = self.box.update_text_input(32, "fist")

        assert result == "fist "

    #Test upper boundary printable key (~) appends its character case.
    def test_update_text_input_tilde_key_appends_character(self):

        result = self.box.update_text_input(126, "fist")

        assert result == "fist~"

    #Test key below the printable range is ignored case.
    def test_update_text_input_below_printable_range_ignored(self):

        result = self.box.update_text_input(31, "fist")

        assert result == "fist"

    #Test key above the printable range is ignored case.
    def test_update_text_input_above_printable_range_ignored(self):

        result = self.box.update_text_input(127, "fist")

        assert result == "fist"

    #Test Tab key does not modify the text case.
    def test_update_text_input_tab_key_ignored(self):

        result = self.box.update_text_input(9, "fist")

        assert result == "fist"

    #Test Escape key does not modify the text case.
    def test_update_text_input_esc_key_ignored(self):

        result = self.box.update_text_input(27, "fist")

        assert result == "fist"

    #Test appending onto empty text case.
    def test_update_text_input_appends_to_empty_text(self):

        result = self.box.update_text_input(ord("f"), "")

        assert result == "f"