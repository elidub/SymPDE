#!/bin/bash
echo "syncing store/P!"

rsync -zaP --ignore-existing eliasd@snellius.surf.nl:thesis/SymPDE/logs/store/P ~/EliasMBA/Projects/Uni/SymPDE/logs/store

echo 'Done!