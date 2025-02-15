import sys


def error_msg_details(error, error_detail: sys = sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_numb = exc_tb.tb_lineno
    error_msg = f"Error occured in script: [{file_name}], line no.: [{line_numb}], message: [{str(error)}]"
    return error_msg


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys = sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_details(error=error_msg, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
