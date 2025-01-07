import inspect
import sys
from pathlib import Path
from typing import Type
import logging

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
from src.build._BasePipe import BasePipe
from src.tuner._BaseHyperparameterGenerator import BaseHyperparameterGenerator

LOG: logging.Logger = logging.getLogger(__name__)


class ObjectChecker:
    def __init__(self) -> None:
        """Checks objects for their compliance with a reference"""
        pass

    def pipeline(self, pipe: Type[BasePipe]) -> Type[BasePipe]:
        """Check pipeline for framework compatibility

        Args:
            pipe (BasePipe): pipeline

        Raises:
            AttributeError: raised if incompatibility found

        Returns:
            BasePipe: unchanged pipeline
        """
        LOG.info("Check User Pipeline '%s' for Framework compatibility", pipe.__class__.__name__)
        # test init base pipe
        ref_pipe = BasePipe()
        test_pipe = pipe()

        # check
        checks_passed = self.__check_class(ref_class=ref_pipe, check_class=test_pipe)

        # summary
        if checks_passed:
            LOG.info("All checks for User Pipeline '%s' passed", pipe.__class__.__name__)
        else:
            LOG.critical("User Pipeline '%s' not compatible with framework", pipe.__class__.__name__)
            raise AttributeError

        return pipe

    def hyperparameter_generator(self, generator: BaseHyperparameterGenerator) -> BaseHyperparameterGenerator:
        """Check hyperparameter generator for framework compatibility

        Args:
            generator (BaseHyperparameterGenerator): hyperparameter generator

        Raises:
            AttributeError: raised if incompatibility found

        Returns:
            BaseHyperparameterGenerator: unchanged hyperparameter generator
        """
        LOG.info("Check Hyperparameter Generator '%s' for Framework compatibility", generator.__class__.__name__)
        # test init base pipe
        ref_generator = BaseHyperparameterGenerator()

        # check
        checks_passed = self.__check_class(ref_class=ref_generator, check_class=generator)

        # summary
        if checks_passed:
            LOG.info("All checks for Hyperparameter Generator '%s' passed", generator.__class__.__name__)
        else:
            LOG.critical("Hyperparameter Generator '%s' not compatible with framework", generator.__class__.__name__)
            raise AttributeError

        return generator

    def __check_class(self, ref_class: object, check_class: object) -> bool:
        """Check if class complies with reference

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        # check __init__
        init_ok = self.__check_passed_init(ref_class=ref_class, check_class=check_class)

        # check methods
        methods_ok = self.__check_passed_methods(ref_class=ref_class, check_class=check_class)

        # check arguments of callable attributes
        arguments_ok = self.__check_passed_callable_arguments(ref_class=ref_class, check_class=check_class)

        return init_ok and methods_ok and arguments_ok

    def __check_passed_init(self, ref_class: object, check_class: object) -> bool:
        """Checks if __init__ signature is sufficient

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        ref_parameters = set(inspect.signature(ref_class.__class__).parameters)
        check_parameters = set(inspect.signature(check_class.__class__).parameters)
        missing_parameters = ref_parameters - check_parameters
        additional_parameters = check_parameters - ref_parameters
        if missing_parameters:
            LOG.critical("Init misses arguments: %s", missing_parameters)
        else:
            LOG.info("Init OK")
        if additional_parameters:
            LOG.warning("Init has additional arguments: %s - Potential undefined behavior", additional_parameters)

        return not missing_parameters

    def __check_passed_methods(self, ref_class: object, check_class: object) -> bool:
        """Checks if all required methods exist

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        ref_attr = set(dir(ref_class))
        check_attr = set(dir(check_class))
        missing_attr = ref_attr - check_attr
        additional_attr = check_attr - ref_attr
        if missing_attr:
            LOG.critical("Missing Attribute(s): %s", missing_attr)
        else:
            LOG.info("Attributes OK")
        if additional_attr:
            LOG.info("Additional attributes: %s", additional_attr)

        return not missing_attr

    def __check_passed_callable_arguments(self, ref_class: object, check_class: object) -> bool:
        """Checks if all methods have the required arguments

        Args:
            ref_class (object): initialized class, used as reference
            check_class (object): initialized class to compare with reference

        Returns:
            bool: True if check passed
        """
        # get missing attributes
        ref_attr = set(dir(ref_class))
        check_attr = set(dir(check_class))
        missing_attr = ref_attr - check_attr

        # check
        missing_inners = set([])
        for attr in dir(ref_class):
            if (
                not (attr.startswith("__") and attr.endswith("__"))
                and attr not in missing_attr
                and callable(ref_class.__getattribute__(attr))
                and callable(check_class.__getattribute__(attr))
            ):
                ref_inner_parameters = set(inspect.signature(ref_class.__getattribute__(attr)).parameters)
                check_inner_parameters = set(inspect.signature(check_class.__getattribute__(attr)).parameters)
                missing_inner_parameters = ref_inner_parameters - check_inner_parameters
                missing_inners |= missing_inner_parameters
                additional_inner_parameters = check_inner_parameters - ref_inner_parameters
                if missing_inner_parameters:
                    LOG.critical("Method '%s' misses arguments: %s", attr, missing_inner_parameters)
                else:
                    LOG.info("Method '%s' OK", attr)
                if additional_inner_parameters:
                    LOG.info("Method '%s' has additional arguments: %s", attr, additional_inner_parameters)

        return not missing_inners
