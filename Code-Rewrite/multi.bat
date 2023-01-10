@echo off

set algorithm=gdumb
set dataset=cifar100
set cpt=20

set seeds[0]=9767
set seeds[1]=2843
set seeds[2]=2214
set seeds[3]=5953
set seeds[4]=2461

echo Configuration
echo Algorithm: %algorithm%
echo Dataset: %dataset%
echo Classes Per Task: %cpt%
echo Start Index: %idx%
echo:

FOR %%s IN (9767, 2843, 2214, 5953, 2461) DO (
    echo Seed: %%s
    echo Executing: python result_generator.py --algorithm %algorithm% --dataset %dataset% --cpt %cpt% --seed %%s
    python result_generator.py --algorithm %algorithm% --dataset %dataset% --cpt %cpt% --seed %%s
    echo:
)

echo Multi-running completed