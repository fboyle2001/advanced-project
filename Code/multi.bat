@echo off

@REM set algorithm=scr
set dataset=cifar10
set cpt=2
set samples=500

echo Configuration
@REM echo Algorithm: %algorithm%
echo Dataset: %dataset%
echo Classes Per Task: %cpt%
echo:

@REM 9767, 2843, 2214, 5953, 2461

FOR %%s IN (2843) DO (
    FOR %%a in (novel_bn, novel_rd) DO (
        echo Seed: %%s
        echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        echo:
    )
)

FOR %%s IN (2214) DO (
    FOR %%a in (ewc, gdumb, l2p, rainbow, scr, novel_bn, novel_rd) DO (
        echo Seed: %%s
        echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        echo:
    )
)

FOR %%s IN (5953, 2461) DO (
    FOR %%a IN (der, derpp, ewc, gdumb, l2p, rainbow, scr, novel_bn, novel_rd) DO (
        echo Seed: %%s
        echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %samples%
        echo:
    )
)

echo Multi-running completed