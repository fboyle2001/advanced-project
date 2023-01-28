@echo off

set algorithm=scr
set dataset=cifar10
set cpt=2

echo Configuration
echo Algorithm: %algorithm%
echo Dataset: %dataset%
echo Classes Per Task: %cpt%
echo:

@REM 9767, 2843, 2214, 5953, 2461

FOR %%s IN (9767, 2843, 2214, 5953, 2461) DO (
    echo Seed: %%s
    echo Executing: python result_generator.py --algorithm %algorithm% --dataset %dataset% --cpt %cpt% --seed %%s
    python result_generator.py --algorithm %algorithm% --dataset %dataset% --cpt %cpt% --seed %%s
    echo:
)

echo Multi-running completed