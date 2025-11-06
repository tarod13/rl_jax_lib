import sys
from dataclasses import dataclass, fields, asdict
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments as key=value pairs.
    Supports both --key value and key=value formats.
    """
    args = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Handle key=value format
        if '=' in arg:
            key, value = arg.split('=', 1)
            key = key.lstrip('-')  # Remove leading dashes
            args[key] = value
            i += 1
        # Handle --key value format
        elif arg.startswith('--'):
            key = arg.lstrip('-')
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                value = sys.argv[i + 1]
                args[key] = value
                i += 2
            else:
                # Flag without value (assume True)
                args[key] = 'True'
                i += 1
        else:
            i += 1
    
    return args


def convert_type(value: str, target_type: type) -> Any:
    """Convert string value to target type."""
    if target_type == bool:
        return value.lower() in ('true', '1', 'yes')
    elif target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == str:
        return value
    else:
        # Handle Optional types
        if hasattr(target_type, '__origin__'):
            if target_type.__origin__ is type(None):
                return None
            # For Union types (like int | None)
            args = getattr(target_type, '__args__', ())
            for arg_type in args:
                if arg_type is not type(None):
                    return convert_type(value, arg_type)
        return value


def create_config_from_args(config_class: type, args: Dict[str, Any]):
    """
    Create a config instance from a dataclass, overriding defaults with provided args.
    """
    # Get all fields from the dataclass
    field_dict = {}
    for field in fields(config_class):
        if field.name in args:
            # Override with provided value
            value = args[field.name]
            # Convert to correct type
            field_dict[field.name] = convert_type(value, field.type)
        # else: use default from dataclass
    
    return config_class(**field_dict)


if __name__ == "__main__":
    # Test
    args = parse_args()
    print("Parsed arguments:")
    for key, value in args.items():
        print(f"  {key} = {value}")