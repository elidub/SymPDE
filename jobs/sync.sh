#!/bin/bash
echo "syncing store/P!"

sync_run_ids=$(cat "$HOME/EliasMBA/Projects/Uni/SymPDE/logs/store/sync_run_ids.txt")

eval rsync -zaP --ignore-existing eliasd@snellius.surf.nl:thesis/SymPDE/logs/store/P/$sync_run_ids.npy  ~/EliasMBA/Projects/Uni/SymPDE/logs/store/P

echo 'Done!'
