{
    'name': 'Meeting Application Backup',
    'description': '发布会议，统计与会人员',
    'author': 'Cao Fa',
    'depends': ['mail'],
    'application': True,
    'data': [
        'meeting_view.xml',
        'security/ir.model.access.csv',
        'security/meeting_access_rules.xml',
    ]
}
