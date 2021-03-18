# path = r'C:\temp\r.csv'
#
# df = pd.read_csv(path).iloc[:,1:]; df.head()
#
# qs = df['date'].str.replace(' ', '-'); qs.head()
#
# df['date'] = pd.to_datetime(qs)
#
# df['date'] = df['date'] + pd.offsets.QuarterEnd(0)
#
# df.set_index('date', inplace=True)
#
# df.index.freq='Q'
#
# df.index
#
# df.plot(figsize=figsize)
#
# cfn = stb.utils.file.create_file_name
#
# df.to_csv(cfn(silent=False))
#
# df = pd.read_csv(r'C:\temp\results_2021_03_18_08_27_58.csv')
#
# df = stb.utils.dframe.to_pd_todatetime(df, 'date')
#
#
#
# df.set_index('date', inplace=True)
#
# df.index.freq = 'Q'
#
# df.index