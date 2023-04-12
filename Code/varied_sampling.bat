@echo off

set dataset=cifar10
set cpt=2

echo Configuration
echo Dataset: %dataset%
echo Classes Per Task: %cpt%
echo:

@REM 9767, 2843, 2214, 5953, 2461
@REM 200, 500, 1000, 2000
@REM scr, rainbow, gdumb, der, derpp, novel

FOR %%s IN (9767) DO (
    FOR %%l IN (2000, 1000, 200) DO (
        FOR %%a IN (novel_rd) DO (
            echo Seed: %%s, Samples %%l 
            echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            echo:
        )
    )
)

echo Multi-running completed