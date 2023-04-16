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

@REM Covers the missing first runs for RD
FOR %%s IN (9767) DO (
    FOR %%l IN (200, 1000, 2000) DO (
        FOR %%a IN (novel_rd) DO (
            echo Seed: %%s, Samples %%l 
            echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            echo:
        )
    )
)


FOR %%s IN (2843, 2214) DO (
    @REM Covers the missing 2nd and 3rd runs for DER, DER++, GDumb, BN, RD, Rainbow, SCR for 0.2, 1, 2k samples
    FOR %%l IN (200, 1000, 2000) DO (
        FOR %%a IN (der, derpp, gdumb, novel_bn, novel_rd, rainbow, scr) DO (
            echo Seed: %%s, Samples %%l     
            echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            echo:
        )
    )

    @REM Covers the missing 2nd and 3rd runs for GDumb, BN, RD, Rainbow for 5k samples
    FOR %%l IN (5000) DO (
        FOR %%a IN (gdumb, novel_bn, novel_rd, rainbow) DO (
            echo Seed: %%s, Samples %%l     
            echo Executing: python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            python result_generator.py --algorithm %%a --dataset %dataset% --cpt %cpt% --seed %%s --samples %%l
            echo:
        )
    )
)

echo Multi-running completed