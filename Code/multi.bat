@echo off

@REM set algorithm=scr
set dataset=cifar100
set cpt=20
set samples=5000

echo Configuration
@REM echo Algorithm: %algorithm%
echo Dataset: %dataset%
echo Classes Per Task: %cpt%
echo:

@REM 9767, 2843, 2214, 5953, 2461

FOR %%s IN (2843, 2214, 5953, 2461) DO (
    FOR %%a in (novel_rd, novel_bn) DO (
        echo Seed: %%s
        echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        echo:
    )
)

echo Multi-running completed