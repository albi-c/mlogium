class ABI:
    @staticmethod
    def namespaced_name(namespace: str, name: str) -> str:
        return f"{name}@{namespace}"

    @staticmethod
    def return_value(function: str) -> str:
        return f"__func_{function}_ret"

    @staticmethod
    def function_end(function: str) -> str:
        return f"__func_{function}_end"

    @staticmethod
    def function_return_address() -> str:
        return f"%__func_ret_addr"

    @staticmethod
    def function_return_value() -> str:
        return f"%__func_ret_val"

    @staticmethod
    def attribute(name: str, attrib: str) -> str:
        return f"{name}.{attrib}"

    @staticmethod
    def static_attribute(name: str, attrib: str) -> str:
        return f"{name}::{attrib}"

    @staticmethod
    def function_parameter(i: int) -> str:
        return f"%__func_param_{i}"

    @staticmethod
    def function_label(function: str) -> str:
        return f"__func_{function}_start"

    @staticmethod
    def label_var(label: str) -> str:
        return f"${label}"
