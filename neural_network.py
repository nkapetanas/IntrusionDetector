
field_names = [ "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

f = open("data/kddcup.data_10_percent_corrected", "r")
logs = []
answers = []
for log in f:
    tokens = log.split(',')
    features = log.split(',')[ :-1]
    #answers.append(tokens[-1]) # -1 means that we take the last element from the list
    #for i in range(0 , len(field_names)):
        #features[i] = str(features[i]) + field_names[i]
    logs.append(features)

for index, answer in enumerate(answers):
    answers[index]= answer.rstrip()
    answers[index] = answers[index].strip('.')

# starting point and ending point for word2vec
for log in logs:
    log.insert(0,"***")
    log.append("###")
    print(log)

# for counting unique values
# #lstemp = [log[22] for log in logs]
#temp = list(set(lstemp))
#print(len(temp))






