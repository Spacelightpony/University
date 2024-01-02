#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class VK_connection: 
    """Class для парсинга данных из VK API""" 
 
    def __init__(self, access_token, api_v: str = '5.154') -> None: 
        self.access_token = access_token 
        self.api_v = api_v
        self.api = vk.API(access_token = self.access_token, v=self.api_v)
 
    """Парсер информации о своем профиле"""
    def get_one_user(self, user_id: int) -> dict: 
        
        return self.api.users.get(user_ids=user_id)
 
    """Парсер пользователей по городу и полу"""
    def get_users(self, city: int, sex: int ) -> pd.DataFrame: 
        
        
        users_list = []
        
        for first_date in tqdm(range(5,95,10)):
              for sort_id in range(0,2):
                    time.sleep(1) #чтобы не было блокировки на запросы
                    df = self.api.users.search(access_token=self.access_token, sort=sort_id, count=1000, city=city, sex=sex, fields = ['bdate'], age_from=first_date, age_to=first_date+10) 
                    df = pd.io.json.json_normalize(df['items'])

                    #удаляем ненужные колонки
                    df = df.drop('track_code', axis=1)
                    df = df.drop('can_access_closed', axis=1)
                    df = df.drop('is_closed', axis=1)

                    #создаем признак - пол
                    if sex == 2:
                        df['sex'] = 'Male'
                    if sex == 1:
                        df['sex'] = 'Female'

                    #преобразовываем дату рождения в возраст
                    now = dt.datetime.now()
                    df['bdate'] = df['bdate'].astype(str)
                    df = df[df['bdate'] != 'nan']
                    df = df[df['bdate'].str.match('^\d{2}.\d{2}.\d{4}$')]
                    df['bdate'] = df['bdate'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))
                    df['age'] = now.year - df['bdate'].dt.year
                    users_list.append(df)
        #соединяем выгрузки в 1 датафрейм
        df = pd.concat(users_list)
        #удаляем дубликаты
        df = df.drop_duplicates(subset=['id'], keep='first')
        #берем в случайном порядке 500 пользователей
        df = df.sample(n=500)
        return df

