ps -ef | grep parallel_minimax_remover.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep parallel_test.sh | grep -v grep | awk '{print $2}' | xargs kill -9